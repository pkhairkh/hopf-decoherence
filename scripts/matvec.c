/*
 * matvec.c - Sparse matrix-vector product C extension for rank computation.
 *
 * Computes matvec (A @ v) and rmatvec (A^* @ v) for a sparse matrix A
 * stored as flat binary files on disk, mmap'd and never fully loaded into RAM.
 *
 * Parallelisation:
 *   matvec  - approach (b): partition by row range. Each thread scans ALL
 *             nnz entries but only accumulates for rows in [r_start, r_end).
 *             Conflict-free, no atomics, no per-thread full-size buffers.
 *   rmatvec - per-thread local buffers (n_cols is small: 9.4M x 16B x 15 = 2.25 GB).
 *
 * Compile:
 *   gcc -O3 -fopenmp -shared -fPIC -o matvec.so matvec.c -lm
 *
 * If data is sorted by row_idx (optional), set sorted=1 via set_sorted()
 * to use binary-search row-range partitioning (O(n_nnz) instead of O(n_nnz * n_threads)).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <complex.h>
#include <omp.h>

/* ==================== Type Definitions ==================== */

typedef float complex  c64_t;    /* complex64  (8 bytes)  */
typedef double complex c128_t;   /* complex128 (16 bytes) */

/*
 * MVHandle: opaque handle returned to Python.
 *
 * All data arrays are mmap'd read-only and shared across threads.
 * The rmatvec_locals are preallocated per-thread buffers for rmatvec
 * (each of size n_cols * sizeof(c128_t)).
 */
typedef struct {
    /* ---- mmap'd data arrays (flattened across all chunks) ---- */
    const int32_t *row_idx;       /* [n_nnz] row indices    (4 B each) */
    const int32_t *cols;          /* [n_nnz] column indices  (4 B each) */
    const c64_t   *vals;          /* [n_nnz] complex64 vals  (8 B each) */

    /* ---- dimensions ---- */
    int64_t n_nnz;                /* total nonzeros           */
    int64_t n_rows;               /* matrix row dimension     */
    int64_t n_cols;               /* matrix column dimension  */

    /* ---- mode ---- */
    int sorted;                   /* 0 = unsorted (approach b), 1 = sorted by row_idx */
    int nthreads;                 /* OpenMP thread count      */

    /* ---- per-thread local buffers for rmatvec ---- */
    c128_t **rmatvec_locals;      /* [nthreads][n_cols]       */

    /* ---- file descriptors and mapping sizes for cleanup ---- */
    int    fd_row_idx, fd_cols, fd_vals;
    size_t sz_row_idx, sz_cols, sz_vals;
} MVHandle;

/* ==================== Helper: mmap a file ==================== */

static void *
map_file(const char *path, int *fd_out, size_t *sz_out)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "matvec: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "matvec: fstat failed for %s: %s\n", path, strerror(errno));
        close(fd);
        return NULL;
    }
    size_t sz = (size_t)st.st_size;
    if (sz == 0) {
        fprintf(stderr, "matvec: file %s is empty\n", path);
        close(fd);
        return NULL;
    }

    /*
     * MAP_SHARED  : changes (none, since PROT_READ) are visible to other processes.
     * MAP_NORESERVE: don't reserve swap space for the mapping (we only read).
     */
    void *p = mmap(NULL, sz, PROT_READ, MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "matvec: mmap failed for %s (%zu bytes): %s\n",
                path, sz, strerror(errno));
        close(fd);
        return NULL;
    }

    *fd_out = fd;
    *sz_out = sz;
    return p;
}

/* ==================== lower_bound (for sorted mode) ==================== */

/*
 * Returns the first index i in [0, n) where arr[i] >= key.
 * Assumes arr is sorted in non-decreasing order.
 */
static inline int64_t
lower_bound_i32(const int32_t *arr, int64_t n, int32_t key)
{
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/* ==================== API: load_matvec_data ==================== */

/*
 * Called from Python via ctypes.
 *
 *   row_idx_path : path to int32[n_nnz] file
 *   cols_path    : path to int32[n_nnz] file
 *   vals_path    : path to complex64[n_nnz] file (8 bytes each)
 *   n_nnz        : total nonzero count
 *   n_rows       : matrix row dimension
 *   n_cols       : matrix column dimension
 *
 * Returns an opaque pointer (cast to MVHandle*), or NULL on failure.
 */
void *
load_matvec_data(const char *row_idx_path,
                 const char *cols_path,
                 const char *vals_path,
                 int64_t n_nnz,
                 int64_t n_rows,
                 int64_t n_cols)
{
    MVHandle *h = (MVHandle *)calloc(1, sizeof(MVHandle));
    if (!h) {
        fprintf(stderr, "matvec: calloc handle failed\n");
        return NULL;
    }

    h->n_nnz  = n_nnz;
    h->n_rows = n_rows;
    h->n_cols = n_cols;
    h->sorted = 0;

    /* ---- mmap row_idx ---- */
    h->row_idx = (const int32_t *)map_file(row_idx_path, &h->fd_row_idx, &h->sz_row_idx);
    if (!h->row_idx) goto fail;

    /* ---- mmap cols ---- */
    h->cols = (const int32_t *)map_file(cols_path, &h->fd_cols, &h->sz_cols);
    if (!h->cols) goto fail;

    /* ---- mmap vals (complex64 = 8 bytes each) ---- */
    h->vals = (const c64_t *)map_file(vals_path, &h->fd_vals, &h->sz_vals);
    if (!h->vals) goto fail;

    /* ---- verify file sizes ---- */
    size_t expect_ri = (size_t)n_nnz * sizeof(int32_t);
    size_t expect_co = (size_t)n_nnz * sizeof(int32_t);
    size_t expect_va = (size_t)n_nnz * sizeof(c64_t);
    if (h->sz_row_idx < expect_ri) {
        fprintf(stderr, "matvec: row_idx file too small (%zu < %zu)\n",
                h->sz_row_idx, expect_ri);
        goto fail;
    }
    if (h->sz_cols < expect_co) {
        fprintf(stderr, "matvec: cols file too small (%zu < %zu)\n",
                h->sz_cols, expect_co);
        goto fail;
    }
    if (h->sz_vals < expect_va) {
        fprintf(stderr, "matvec: vals file too small (%zu < %zu)\n",
                h->sz_vals, expect_va);
        goto fail;
    }

    /* ---- set thread count ---- */
    h->nthreads = omp_get_max_threads();
    if (h->nthreads < 1) h->nthreads = 1;

    /* ---- allocate per-thread local buffers for rmatvec ---- */
    h->rmatvec_locals = (c128_t **)calloc((size_t)h->nthreads, sizeof(c128_t *));
    if (!h->rmatvec_locals) {
        fprintf(stderr, "matvec: calloc rmatvec_locals ptr array failed\n");
        goto fail;
    }
    for (int t = 0; t < h->nthreads; t++) {
        h->rmatvec_locals[t] = (c128_t *)calloc((size_t)n_cols, sizeof(c128_t));
        if (!h->rmatvec_locals[t]) {
            fprintf(stderr, "matvec: calloc rmatvec_locals[%d] failed "
                    "(need %zu bytes for n_cols=%ld)\n",
                    t, (size_t)n_cols * sizeof(c128_t), (long)n_cols);
            /* free already-allocated buffers */
            for (int t2 = 0; t2 < t; t2++) free(h->rmatvec_locals[t2]);
            goto fail;
        }
    }

    fprintf(stderr, "matvec: loaded nnz=%ld n_rows=%ld n_cols=%ld nthreads=%d "
            "(locals: %d x %ld x %zu = %.1f GB)\n",
            (long)n_nnz, (long)n_rows, (long)n_cols, h->nthreads,
            h->nthreads, (long)n_cols, sizeof(c128_t),
            (double)h->nthreads * n_cols * sizeof(c128_t) / 1e9);

    return (void *)h;

fail:
    if (h->row_idx) { munmap((void *)h->row_idx, h->sz_row_idx); close(h->fd_row_idx); }
    if (h->cols)    { munmap((void *)h->cols,    h->sz_cols);    close(h->fd_cols);    }
    if (h->vals)    { munmap((void *)h->vals,    h->sz_vals);    close(h->fd_vals);    }
    free(h->rmatvec_locals);
    free(h);
    return NULL;
}

/* ==================== API: free_matvec_data ==================== */

void
free_matvec_data(void *handle)
{
    MVHandle *h = (MVHandle *)handle;
    if (!h) return;

    if (h->rmatvec_locals) {
        for (int t = 0; t < h->nthreads; t++) {
            free(h->rmatvec_locals[t]);
        }
        free(h->rmatvec_locals);
    }
    if (h->row_idx) { munmap((void *)h->row_idx, h->sz_row_idx); close(h->fd_row_idx); }
    if (h->cols)    { munmap((void *)h->cols,    h->sz_cols);    close(h->fd_cols);    }
    if (h->vals)    { munmap((void *)h->vals,    h->sz_vals);    close(h->fd_vals);    }
    free(h);
}

/* ==================== API: prefault_data ==================== */

/*
 * Touches every page of the mmap'd arrays to force the kernel to load
 * them into the page cache. Call this once after load_matvec_data()
 * to avoid page-fault stalls during the first matvec/rmatvec call.
 *
 * This is sequential and single-threaded (the OS read-ahead handles it).
 * For 64 GB of data on SSD (~1 GB/s), this takes ~64 seconds.
 */
void
prefault_data(void *handle)
{
    MVHandle *h = (MVHandle *)handle;
    if (!h) return;

    const size_t page = 4096;
    volatile int64_t sink = 0;  /* prevent optimiser from eliding reads */

    /* Prefault row_idx */
    for (size_t off = 0; off < h->sz_row_idx; off += page)
        sink += h->row_idx[off / sizeof(int32_t)];

    /* Prefault cols */
    for (size_t off = 0; off < h->sz_cols; off += page)
        sink += h->cols[off / sizeof(int32_t)];

    /* Prefault vals */
    for (size_t off = 0; off < h->sz_vals; off += page)
        sink += (int64_t)h->vals[off / sizeof(c64_t)];

    (void)sink;
}

/* ==================== API: set_sorted ==================== */

/*
 * Tell the handle whether the data is sorted by row_idx.
 * If sorted=1, matvec() will use binary-search row-range partitioning
 * (O(n_nnz) total work instead of O(n_nnz * n_threads)).
 *
 * Call this after sorting the (row_idx, cols, vals) arrays by row_idx.
 */
void
set_sorted(void *handle, int sorted)
{
    MVHandle *h = (MVHandle *)handle;
    if (h) h->sorted = sorted ? 1 : 0;
}

/* ==================== API: getters ==================== */

int64_t get_n_nnz(void  *handle) { MVHandle *h = (MVHandle *)handle; return h ? h->n_nnz  : 0; }
int64_t get_n_rows(void *handle) { MVHandle *h = (MVHandle *)handle; return h ? h->n_rows : 0; }
int64_t get_n_cols(void *handle) { MVHandle *h = (MVHandle *)handle; return h ? h->n_cols : 0; }

/* ==================== matvec (unsorted): approach (b) ==================== */

/*
 * result = A @ v
 *
 * v:      complex128 [n_cols]  (input)
 * result: complex128 [n_rows]  (zeroed and filled)
 *
 * Approach (b): partition by row range.
 * Each thread handles rows [r_start, r_end).
 * It scans ALL n_nnz entries but only accumulates for entries whose
 * row_idx falls in [r_start, r_end).
 *
 * This is O(n_nnz * n_threads) total work, but:
 *   - Conflict-free (no two threads write the same result[r])
 *   - No atomics needed
 *   - No per-thread full-size buffers (just the shared result array)
 *   - Streaming read of row_idx is cache-friendly
 *   - Only 1/n_threads of entries trigger the expensive cols/vals read
 *
 * The sequential scan of row_idx is the bottleneck. For 4B entries
 * with 15 threads, each thread streams through 16 GB of row_idx data.
 */
static void
matvec_unsorted(MVHandle *h, const c128_t *v, c128_t *result)
{
    int64_t n_nnz  = h->n_nnz;
    int64_t n_rows = h->n_rows;
    const int32_t *row_idx = h->row_idx;
    const int32_t *cols    = h->cols;
    const c64_t   *vals    = h->vals;

    /* Zero the result (parallelised for the 25 GB array) */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_rows; i++)
        result[i] = 0;

    #pragma omp parallel
    {
        int tid      = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int64_t r_start = (n_rows * (int64_t)tid)     / nthreads;
        int64_t r_end   = (n_rows * (int64_t)(tid+1)) / nthreads;

        /*
         * We iterate over all entries. For each entry, we read row_idx[i]
         * (sequential, streaming). If the row falls in our range, we also
         * read cols[i], vals[i], v[cols[i]], and accumulate into result[r].
         *
         * The branch predictor will mispredict ~1/n_threads of the time,
         * but the streaming read of row_idx is the real bottleneck.
         */
        for (int64_t i = 0; i < n_nnz; i++) {
            int32_t r = row_idx[i];
            if (r >= r_start && r < r_end) {
                result[r] += (c128_t)vals[i] * v[cols[i]];
            }
        }
    }
}

/* ==================== matvec (sorted): binary-search partitioning ==================== */

/*
 * Same interface as matvec_unsorted, but assumes row_idx is sorted.
 *
 * Each thread binary-searches for the entry range [start, end) where
 * row_idx[start] >= r_start and row_idx[end-1] < r_end.
 * Then it iterates only over [start, end) — no redundant scanning.
 *
 * Total work: O(n_nnz) + O(n_threads * log(n_nnz)).
 * This is 15x faster than the unsorted version for 15 threads.
 */
static void
matvec_sorted_impl(MVHandle *h, const c128_t *v, c128_t *result)
{
    int64_t n_nnz  = h->n_nnz;
    int64_t n_rows = h->n_rows;
    const int32_t *row_idx = h->row_idx;
    const int32_t *cols    = h->cols;
    const c64_t   *vals    = h->vals;

    /* Zero the result */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_rows; i++)
        result[i] = 0;

    #pragma omp parallel
    {
        int tid      = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int64_t r_start = (n_rows * (int64_t)tid)     / nthreads;
        int64_t r_end   = (n_rows * (int64_t)(tid+1)) / nthreads;

        /* Binary search for the first entry with row_idx >= r_start */
        int64_t start = lower_bound_i32(row_idx, n_nnz, (int32_t)r_start);

        /* Binary search for the first entry with row_idx >= r_end */
        int64_t end = (tid == nthreads - 1)
            ? n_nnz
            : lower_bound_i32(row_idx, n_nnz, (int32_t)r_end);

        /* Process only [start, end) — no conflicts since rows don't overlap */
        for (int64_t i = start; i < end; i++) {
            result[row_idx[i]] += (c128_t)vals[i] * v[cols[i]];
        }
    }
}

/* ==================== API: matvec (dispatcher) ==================== */

/*
 * result = A @ v
 *   v      : pointer to complex128 [n_cols]
 *   result : pointer to complex128 [n_rows] (will be zeroed and filled)
 */
void
matvec(void *handle, const void *v_ptr, void *result_ptr)
{
    MVHandle *h = (MVHandle *)handle;
    const c128_t *v = (const c128_t *)v_ptr;
    c128_t *result   = (c128_t *)result_ptr;

    if (h->sorted)
        matvec_sorted_impl(h, v, result);
    else
        matvec_unsorted(h, v, result);
}

/* ==================== API: rmatvec ==================== */

/*
 * result = A^* @ v   (conjugate transpose)
 *   v      : pointer to complex128 [n_rows]
 *   result : pointer to complex128 [n_cols] (will be zeroed and filled)
 *
 * Formula:
 *   result[c] = sum_{i: cols[i]=c} conj(vals[i]) * v[row_idx[i]]
 *
 * The scatter target is result[cols[i]], which has only n_cols = 9.4M
 * unique values. We use per-thread local buffers (150 MB each, 2.25 GB
 * total for 15 threads) and then reduce.
 *
 * The #pragma omp for splits the entry range [0, n_nnz) among threads.
 * Each thread writes to its own local buffer at local[cols[i]] — no conflicts.
 *
 * Memory access pattern per thread:
 *   - Read row_idx[i] : sequential (streaming)
 *   - Read cols[i]    : sequential (streaming)
 *   - Read vals[i]    : sequential (streaming)
 *   - Read v[row_idx[i]] : RANDOM (25 GB array, ~4B/15 random reads)
 *   - Write local[cols[i]] : random within 150 MB (fits partially in L3)
 *
 * The random read of v[row_idx[i]] is the bottleneck for unsorted data.
 * If data is sorted by row_idx, v[row_idx[i]] becomes sequential — 4x faster.
 */
void
rmatvec(void *handle, const void *v_ptr, void *result_ptr)
{
    MVHandle *h = (MVHandle *)handle;
    const c128_t *v = (const c128_t *)v_ptr;   /* v: [n_rows] */
    c128_t *result   = (c128_t *)result_ptr;    /* result: [n_cols] */

    int64_t n_nnz  = h->n_nnz;
    int64_t n_cols = h->n_cols;
    const int32_t *row_idx = h->row_idx;
    const int32_t *cols    = h->cols;
    const c64_t   *vals    = h->vals;
    int nthreads = h->nthreads;

    /* ---- zero the per-thread local buffers ---- */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        memset(h->rmatvec_locals[tid], 0, (size_t)n_cols * sizeof(c128_t));
    }

    /* ---- parallel scatter-add into per-thread locals ---- */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        c128_t *local = h->rmatvec_locals[tid];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n_nnz; i++) {
            /* conj(vals[i]) * v[row_idx[i]]  accumulated into local[cols[i]] */
            local[cols[i]] += conj((c128_t)vals[i]) * v[row_idx[i]];
        }
    }

    /* ---- reduction: sum all local buffers into result ---- */
    c128_t **locals = h->rmatvec_locals;

    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < n_cols; j++) {
        c128_t sum = locals[0][j];
        for (int t = 1; t < nthreads; t++) {
            sum += locals[t][j];
        }
        result[j] = sum;
    }
}

/* ==================== API: compute_column_norms ==================== */

/*
 * Computes the column norms of A (diagonal of A^* A).
 * Useful as a Jacobi preconditioner for LOBPCG.
 *
 * result[j] = sum_{i: cols[i]=j} |vals[i]|^2
 *
 * Uses the same per-thread local buffer strategy as rmatvec.
 * result: double [n_cols]
 */
void
compute_column_norms(void *handle, void *result_ptr)
{
    MVHandle *h = (MVHandle *)handle;
    double *result = (double *)result_ptr;

    int64_t n_nnz  = h->n_nnz;
    int64_t n_cols = h->n_cols;
    const int32_t *cols = h->cols;
    const c64_t   *vals = h->vals;
    int nthreads = h->nthreads;

    /* Use the first nthreads local buffers, reinterpreted as double arrays.
     * Each c128_t local is 16 bytes; we use the real part only.
     * Actually, we need separate double buffers. Let's use the real parts
     * of the c128_t locals (the imaginary part will stay 0). */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *local = (double *)h->rmatvec_locals[tid];  /* reuse as double[n_cols*2] */
        memset(local, 0, (size_t)n_cols * sizeof(double));

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n_nnz; i++) {
            float re = crealf(vals[i]);
            float im = cimagf(vals[i]);
            local[cols[i]] += (double)(re * re + im * im);
        }
    }

    /* Reduction */
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < n_cols; j++) {
        double sum = 0;
        for (int t = 0; t < nthreads; t++) {
            sum += ((double *)h->rmatvec_locals[t])[j];
        }
        result[j] = sum;
    }
}

/* ==================== End of file ==================== */
