/*
 * rank_solver.c -- Compute the nullity of a large sparse complex matrix
 *                  via randomized subspace iteration with spectral shift.
 *
 * ALGORITHM OVERVIEW
 * ==================
 *
 * The nullity of A equals the number of zero eigenvalues of A^*A
 * (since ||Av||^2 = v^*A^*Av, so Av=0  iff  A^*Av=0).
 *
 * We cannot form A^*A explicitly (9.4M x 9.4M), but we can apply it
 * via two sparse matvecs:  A^*A v = A^*(A v).
 *
 * To find the SMALLEST eigenvalues of A^*A (including the zeros),
 * we use subspace iteration on the shifted matrix
 *
 *     B = c*I - A^*A,        c > ||A||^2 (spectral norm squared).
 *
 * The eigenvalues of B are:
 *   - c        (multiplicity = nullity,  for eigenvectors in null(A))
 *   - c - s_i^2 (< c,  for all other singular values s_i of A)
 *
 * Subspace iteration on B converges to the eigenvectors with the
 * LARGEST eigenvalues of B, i.e. the SMALLEST eigenvalues of A^*A --
 * precisely the null space of A.
 *
 * After q iterations, we form the k x k Ritz matrix G = Q^* A^*A Q
 * and compute its eigenvalues.  The number of Ritz values near zero
 * is the estimated nullity.
 *
 * Convergence rate per iteration:
 *   |1 - s_{d+1}^2 / c|^q    (d = nullity, s_{d+1} = smallest nonzero sing. val.)
 *
 * With c ~ 1.05 * ||A||^2 and a reasonable spectral gap (s_{d+1}/||A||
 * not too small), 20--40 iterations suffice.
 *
 * REFERENCES
 * ==========
 * [1] N. Halko, P.-G. Martinsson, J. A. Tropp,
 *     "Finding Structure with Randomness: Probabilistic Algorithms for
 *      Constructing Approximate Matrix Decompositions,"
 *     SIAM Review 53(2):217-288, 2011.  arXiv:0909.4061
 *
 * [2] M. Gu, "Subspace Iteration Randomization and Singular Value
 *     Problems," SIAM J. Numer. Anal. 54(2):949-973, 2016.
 *     arXiv:1408.2208
 *
 * [3] D. Kressner, "A randomized small-block Lanczos method for
 *     large-scale null space computation," 2024.  arXiv:2407.04634
 *
 * [4] E. Polizzi, "Density-matrix-based algorithm for solving eigenvalue
 *     problems," Phys. Rev. B 79:115112, 2009.
 *
 * [5] T. Sakurai, H. Sugiura, "A projection method for generalized
 *     eigenvalue problems," J. Comput. Appl. Math. 159:119-128, 2003.
 *
 * BUILD
 * =====
 *   gcc -O3 -fopenmp -o rank_solver rank_solver.c \
 *       -lm -llapack -lblas -lgfortran
 *
 *   (On the target machine with OpenBLAS, replace -lblas with -lopenblas)
 *
 * USAGE
 * =====
 *   ./rank_solver [row_idx.bin] [cols.bin] [vals.bin] [k] [iters] [tol_exp]
 *
 *   Defaults:  k=20, iters=30, tol_exp=-8
 *
 * MEMORY BUDGET (k=20)
 * ====================
 *   row_idx mmap:   16.4 GB
 *   cols    mmap:   16.4 GB
 *   vals    mmap:   32.7 GB
 *   Q (n_cols*k):    3.0 GB
 *   W (n_cols*k):    3.0 GB
 *   Y (n_rows):     25.1 GB
 *   local bufs:      2.3 GB   (n_threads * n_cols * 16)
 *   Misc:           <0.1 GB
 *   Total:         ~99 GB  (fits in 125 GB)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

/* ====================================================================== */
/*  Type definitions                                                      */
/* ====================================================================== */

typedef double _Complex dcomplex;   /* 16 bytes */
typedef float  _Complex scomplex;   /*  8 bytes */

/* ====================================================================== */
/*  Matrix dimensions                                                     */
/* ====================================================================== */

static const int64_t N_ROWS = 1568987626LL;
static const int64_t N_COLS = 9448324LL;
static const int64_t NNZ    = 4091264798LL;

/* ====================================================================== */
/*  LAPACK / BLAS prototypes (Fortran calling convention)                */
/* ====================================================================== */

/* QR factorization:  Q = QR(A)  */
extern void zgeqrf_(const int *m, const int *n, dcomplex *a,
                    const int *lda, dcomplex *tau,
                    dcomplex *work, const int *lwork, int *info);

/* Generate Q from QR factorization */
extern void zungqr_(const int *m, const int *n, const int *k,
                    dcomplex *a, const int *lda, const dcomplex *tau,
                    dcomplex *work, const int *lwork, int *info);

/* Hermitian eigenvalues (divide-and-conquer) */
extern void zheevd_(const char *jobz, const char *uplo, const int *n,
                    dcomplex *a, const int *lda, double *w,
                    dcomplex *work, const int *lwork,
                    double *rwork, const int *lrwork,
                    int *iwork, const int *liwork, int *info);

/* Complex matrix-matrix multiply:  C = alpha*op(A)*op(B) + beta*C */
extern void zgemm_(const char *transa, const char *transb,
                   const int *m, const int *n, const int *k,
                   const dcomplex *alpha,
                   const dcomplex *a, const int *lda,
                   const dcomplex *b, const int *ldb,
                   const dcomplex *beta,
                    dcomplex *c, const int *ldc);

/* ====================================================================== */
/*  PRNG: xorshift128+  (no external dependency)                         */
/* ====================================================================== */

typedef struct { uint64_t s[2]; } prng_t;

static inline uint64_t prng_u64(prng_t *p)
{
    uint64_t s1 = p->s[0];
    uint64_t s0 = p->s[1];
    uint64_t result = s0 + s1;
    p->s[0] = s0;
    s1 ^= s1 << 23;
    p->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return result;
}

/* Uniform double in (0, 1) */
static inline double prng_uniform(prng_t *p)
{
    return ((double)(prng_u64(p) >> 11) + 1.0) * (1.0 / 9007199254740992.0);
}

/* Standard normal via Box-Muller */
static inline double prng_normal(prng_t *p)
{
    double u1 = prng_uniform(p);
    double u2 = prng_uniform(p);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/*
 * Complex Gaussian: real and imaginary parts are independent N(0, 1/sqrt(2)).
 * This ensures E[|z|^2] = 1, making the random subspace isotropic.
 */
static inline dcomplex prng_cnormal(prng_t *p)
{
    double u1 = prng_uniform(p);
    double u2 = prng_uniform(p);
    double r  = sqrt(-log(u1));
    double re = r * cos(2.0 * M_PI * u2);
    double im = r * sin(2.0 * M_PI * u2);
    return (re + I * im) * M_SQRT1_2;
}

/* ====================================================================== */
/*  Aligned allocation helper                                             */
/* ====================================================================== */

static void *xaligned_alloc(size_t alignment, size_t size)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "Fatal: allocation of %zu bytes failed\n", size);
        exit(1);
    }
    return ptr;
}

/* ====================================================================== */
/*  mmap helper                                                           */
/* ====================================================================== */

static void *mmap_file(const char *path, size_t size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }

    void *ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); close(fd); exit(1); }

    /* Hint the kernel: we will read sequentially. */
    (void)madvise(ptr, size, MADV_SEQUENTIAL);
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    /* We keep the fd open so the mapping stays valid. */
    return ptr;
}

/* ====================================================================== */
/*  matvec:  result = A @ v                                               */
/*                                                                         */
/*  Uses OpenMP atomics on the scatter-add.  Each entry is processed      */
/*  exactly once (no redundant scanning).  The real and imaginary parts   */
/*  of each result element are updated with separate atomic adds.         */
/*                                                                         */
/*  v:      dcomplex[N_COLS]   (input)                                    */
/*  result: dcomplex[N_ROWS]   (output, zeroed internally)               */
/* ====================================================================== */

static void matvec(const int32_t * restrict row_idx,
                   const int32_t * restrict cols,
                   const scomplex * restrict vals,
                   const dcomplex * restrict v,
                   dcomplex * restrict result,
                   int n_threads)
{
    double * restrict rd = (double * restrict)result;

    /* Zero result (parallel memset) */
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t i = 0; i < 2LL * N_ROWS; i++)
        rd[i] = 0.0;

    /* Scatter-add with atomics */
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t e = 0; e < NNZ; e++) {
        int32_t row = row_idx[e];
        int32_t col = cols[e];

        /* Upcast float-complex to double */
        double vr = crealf(vals[e]);
        double vi = cimagf(vals[e]);

        double wr = creal(v[col]);
        double wi = cimag(v[col]);

        /* Complex multiply: (vr+i*vi) * (wr+i*wi) */
        double pr = vr * wr - vi * wi;
        double pi = vr * wi + vi * wr;

        /* Atomic add (real and imaginary parts are independent) */
        int64_t idx = 2LL * (int64_t)row;
        #pragma omp atomic
        rd[idx]   += pr;
        #pragma omp atomic
        rd[idx+1] += pi;
    }
}

/* ====================================================================== */
/*  rmatvec:  result = A^* @ v    (conjugate transpose)                  */
/*                                                                         */
/*  Entry-range partitioning with per-thread local buffers of size       */
/*  N_COLS.  After accumulation, the local buffers are reduced into the   */
/*  global result.  No atomics needed.                                    */
/*                                                                         */
/*  v:          dcomplex[N_ROWS]   (input)                                */
/*  result:     dcomplex[N_COLS]   (output)                               */
/*  local_bufs: dcomplex[n_threads * N_COLS]  (scratch)                  */
/* ====================================================================== */

static void rmatvec(const int32_t * restrict row_idx,
                    const int32_t * restrict cols,
                    const scomplex * restrict vals,
                    const dcomplex * restrict v,
                    dcomplex * restrict result,
                    dcomplex * restrict local_bufs,
                    int n_threads)
{
    /* Zero result */
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t c = 0; c < N_COLS; c++)
        result[c] = 0.0;

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        dcomplex * restrict local = local_bufs + (int64_t)tid * N_COLS;

        /* Zero local buffer */
        for (int64_t c = 0; c < N_COLS; c++)
            local[c] = 0.0;

        /* Accumulate: local[col] += conj(val) * v[row] */
        #pragma omp for schedule(static) nowait
        for (int64_t e = 0; e < NNZ; e++) {
            int32_t row = row_idx[e];
            int32_t col = cols[e];
            dcomplex val = (dcomplex)vals[e];   /* upcast */
            local[col] += conj(val) * v[row];
        }

        /* Ensure all threads finished accumulation before reduction */
        #pragma omp barrier

        /* Reduce local buffers -> result */
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < N_COLS; c++) {
            dcomplex sum = 0.0;
            for (int t = 0; t < n_threads; t++)
                sum += local_bufs[(int64_t)t * N_COLS + c];
            result[c] = sum;
        }
    }
}

/* ====================================================================== */
/*  Orthonormalize columns of Q via QR (LAPACK zgeqrf + zungqr)          */
/*                                                                         */
/*  Q is stored column-major:  Q[j*N_COLS + i]  =  Q_{i,j}               */
/*  On exit, columns of Q are orthonormal.                                */
/* ====================================================================== */

static void orthonormalize(dcomplex *Q, int k)
{
    int m = (int)N_COLS;   /* rows    */
    int n = k;              /* columns */
    int lda = m;
    int info;

    dcomplex *tau = (dcomplex *)malloc((size_t)k * sizeof(dcomplex));
    if (!tau) { fprintf(stderr, "malloc tau failed\n"); exit(1); }

    /* Workspace query */
    int lwork = -1;
    dcomplex wkopt;
    zgeqrf_(&m, &n, Q, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) { fprintf(stderr, "zgeqrf query failed: %d\n", info); exit(1); }
    lwork = (int)creal(wkopt);
    if (lwork < 1) lwork = 1;
    dcomplex *work = (dcomplex *)malloc((size_t)lwork * sizeof(dcomplex));
    if (!work) { fprintf(stderr, "malloc work failed\n"); exit(1); }

    /* QR factorization */
    zgeqrf_(&m, &n, Q, &lda, tau, work, &lwork, &info);
    if (info != 0) { fprintf(stderr, "zgeqrf failed: %d\n", info); exit(1); }

    /* Generate Q */
    zungqr_(&m, &n, &n, Q, &lda, tau, work, &lwork, &info);
    if (info != 0) { fprintf(stderr, "zungqr failed: %d\n", info); exit(1); }

    free(work);
    free(tau);
}

/* ====================================================================== */
/*  compute_gram:  G = Q^* W     (k x k Hermitian)                       */
/*                                                                         */
/*  Uses BLAS zgemm.  Q and W are N_COLS x k (column-major).             */
/*  G is k x k (column-major).                                            */
/* ====================================================================== */

static void compute_gram(const dcomplex *Q, const dcomplex *W,
                         dcomplex *G, int k)
{
    char   transa = 'C';            /* conjugate transpose of Q */
    char   transb = 'N';            /* no transpose of W        */
    int    m = k, n = k;
    int    k_blas = (int)N_COLS;   /* inner dimension          */
    int    lda = (int)N_COLS;
    int    ldb = (int)N_COLS;
    int    ldc = k;
    dcomplex alpha = 1.0;
    dcomplex beta  = 0.0;

    zgemm_(&transa, &transb, &m, &n, &k_blas,
           &alpha, Q, &lda, W, &ldb, &beta, G, &ldc);
}

/* ====================================================================== */
/*  eig_hermitian: eigenvalues of a k x k Hermitian matrix               */
/*                                                                         */
/*  On exit, eigenvalues[] are sorted in ASCENDING order.                */
/*  The input matrix G is destroyed.                                      */
/* ====================================================================== */

static void eig_hermitian(dcomplex *G, int k, double *eigenvalues)
{
    char jobz  = 'N';   /* eigenvalues only */
    char uplo  = 'U';   /* upper triangle stored */
    int  n     = k;
    int  lda   = k;
    int  info;

    /* Workspace query */
    int lwork = -1, lrwork = -1, liwork = -1;
    dcomplex wkopt;
    double   rwkopt;
    int      iwkopt;

    zheevd_(&jobz, &uplo, &n, G, &lda, eigenvalues,
            &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info);
    if (info != 0) { fprintf(stderr, "zheevd query failed: %d\n", info); exit(1); }

    lwork  = (int)creal(wkopt);
    lrwork = (int)rwkopt;
    liwork = iwkopt;
    if (lwork  < 1) lwork  = 1;
    if (lrwork < 1) lrwork = 1;
    if (liwork < 1) liwork = 1;

    dcomplex *work  = (dcomplex *)malloc((size_t)lwork  * sizeof(dcomplex));
    double   *rwork = (double   *)malloc((size_t)lrwork * sizeof(double));
    int      *iwork = (int      *)malloc((size_t)liwork * sizeof(int));
    if (!work || !rwork || !iwork) {
        fprintf(stderr, "zheevd workspace allocation failed\n");
        exit(1);
    }

    zheevd_(&jobz, &uplo, &n, G, &lda, eigenvalues,
            work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    if (info != 0) { fprintf(stderr, "zheevd failed: %d\n", info); exit(1); }

    free(work);
    free(rwork);
    free(iwork);
}

/* ====================================================================== */
/*  Helper: apply A^*A to a single column of Q, store result in W[:,j]    */
/* ====================================================================== */

static void apply_ata_column(const int32_t *row_idx,
                             const int32_t *cols,
                             const scomplex *vals,
                             const dcomplex *Q, dcomplex *W,
                             dcomplex *Y,          /* scratch, N_ROWS */
                             dcomplex *R,          /* unused here, kept for API symmetry */
                             dcomplex *local_bufs,
                             int n_threads,
                             int j)
{
    (void)R;  /* R is not used in this function; rmatvec uses local_bufs */
    const dcomplex *qj = Q + (int64_t)j * N_COLS;
    dcomplex *wj = W + (int64_t)j * N_COLS;

    /* Y = A @ qj  (matvec) */
    matvec(row_idx, cols, vals, qj, Y, n_threads);

    /* wj = A^* @ Y  (rmatvec) */
    rmatvec(row_idx, cols, vals, Y, wj, local_bufs, n_threads);
}

/* ====================================================================== */
/*  Helper: compute Ritz values of A^*A restricted to span(Q)            */
/*          W = A^*A Q,  G = Q^* W,  lambda = eig(G)                     */
/* ====================================================================== */

static void compute_ritz(const int32_t *row_idx,
                         const int32_t *cols,
                         const scomplex *vals,
                         const dcomplex *Q,
                         dcomplex *W,          /* output: A^*A Q  */
                         dcomplex *G,          /* scratch k*k     */
                         double *eigenvalues,  /* output          */
                         dcomplex *Y,          /* scratch N_ROWS  */
                         dcomplex *R,          /* scratch N_COLS  */
                         dcomplex *local_bufs,
                         int n_threads,
                         int k)
{
    /* W = A^*A Q  (column by column) */
    for (int j = 0; j < k; j++)
        apply_ata_column(row_idx, cols, vals, Q, W, Y, R,
                         local_bufs, n_threads, j);

    /* G = Q^* W  (k x k) */
    compute_gram(Q, W, G, k);

    /* eigenvalues of Hermitian G */
    eig_hermitian(G, k, eigenvalues);
}

/* ====================================================================== */
/*  Main                                                                  */
/* ====================================================================== */

int main(int argc, char *argv[])
{
    /* ---- Parse arguments ---- */
    const char *row_idx_path = "row_idx_flat.bin";
    const char *cols_path    = "cols_flat.bin";
    const char *vals_path    = "vals_flat.bin";
    /* Defaults: k=10 (nullity 3-4 + 6 margin), iters=40, tol_exp=-6
     *
     * IMPORTANT: k must be SMALL (just above expected nullity) for fast
     * convergence.  See the convergence analysis in the header comment.
     * The user's original suggestion of k=50-100 would give very slow
     * convergence because the gap between consecutive eigenvalues of B
     * shrinks as k increases.
     */
    int  k        = 10;
    int  iters    = 40;
    double tol_exp = -6.0;

    if (argc > 1) row_idx_path = argv[1];
    if (argc > 2) cols_path    = argv[2];
    if (argc > 3) vals_path    = argv[3];
    if (argc > 4) k            = atoi(argv[4]);
    if (argc > 5) iters        = atoi(argv[5]);
    if (argc > 6) tol_exp      = atof(argv[6]);

    int n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;

    double T0 = omp_get_wtime();

    printf("========================================================\n");
    printf("          Sparse Matrix Nullity Solver\n");
    printf("  Randomized Subspace Iteration + Spectral Shift\n");
    printf("========================================================\n\n");
    printf("Matrix:  %lld rows x %lld cols,  NNZ = %lld\n",
           (long long)N_ROWS, (long long)N_COLS, (long long)NNZ);
    printf("Density: %.2e\n", (double)NNZ / ((double)N_ROWS * N_COLS));
    printf("Params:  k=%d  iters=%d  tol=10^(%.1f)*lambda_max  threads=%d\n\n",
           k, iters, tol_exp, n_threads);
    fflush(stdout);

    /* ================================================================
     *  Step 1: mmap the three binary files
     * ================================================================ */
    printf("[1/8] mmap'ing data files ...\n"); fflush(stdout);

    size_t sz_idx  = (size_t)NNZ * sizeof(int32_t);
    size_t sz_cols = (size_t)NNZ * sizeof(int32_t);
    size_t sz_vals = (size_t)NNZ * sizeof(scomplex);

    const int32_t * restrict row_idx = (const int32_t *)mmap_file(row_idx_path, sz_idx);
    const int32_t * restrict cols    = (const int32_t *)mmap_file(cols_path,    sz_cols);
    const scomplex * restrict vals   = (const scomplex *)mmap_file(vals_path,   sz_vals);

    printf("      row_idx : %p  (%.2f GB)\n", (void*)row_idx, sz_idx  / 1e9);
    printf("      cols    : %p  (%.2f GB)\n", (void*)cols,    sz_cols / 1e9);
    printf("      vals    : %p  (%.2f GB)\n", (void*)vals,    sz_vals / 1e9);
    printf("      Total mmap: %.2f GB\n\n", (sz_idx + sz_cols + sz_vals) / 1e9);
    fflush(stdout);

    /* ================================================================
     *  Step 2: Allocate working buffers
     * ================================================================ */
    printf("[2/8] Allocating buffers ...\n"); fflush(stdout);

    /* Q, W: N_COLS x k, column-major, complex128 */
    dcomplex *Q = (dcomplex *)xaligned_alloc(64, (size_t)N_COLS * k * sizeof(dcomplex));
    dcomplex *W = (dcomplex *)xaligned_alloc(64, (size_t)N_COLS * k * sizeof(dcomplex));

    /* Y: N_ROWS complex128 (single matvec result) */
    dcomplex *Y = (dcomplex *)xaligned_alloc(64, (size_t)N_ROWS * sizeof(dcomplex));

    /* R: N_COLS complex128 (rmatvec scratch for power iteration) */
    dcomplex *R = (dcomplex *)xaligned_alloc(64, (size_t)N_COLS * sizeof(dcomplex));

    /* Per-thread local buffers for rmatvec */
    dcomplex *local_bufs = (dcomplex *)xaligned_alloc(64,
        (size_t)n_threads * N_COLS * sizeof(dcomplex));

    /* Small buffers */
    dcomplex *G = (dcomplex *)xaligned_alloc(64, (size_t)k * k * sizeof(dcomplex));
    double   *eigenvalues = (double *)malloc((size_t)k * sizeof(double));

    double mem_gb = ((double)N_COLS * k * 2 + N_ROWS + N_COLS +
                     (double)n_threads * N_COLS) * sizeof(dcomplex) / 1e9;
    printf("      Q      : %.2f GB\n", (double)N_COLS * k * sizeof(dcomplex) / 1e9);
    printf("      W      : %.2f GB\n", (double)N_COLS * k * sizeof(dcomplex) / 1e9);
    printf("      Y      : %.2f GB\n", (double)N_ROWS * sizeof(dcomplex) / 1e9);
    printf("      locals : %.2f GB\n", (double)n_threads * N_COLS * sizeof(dcomplex) / 1e9);
    printf("      Total buffers: %.2f GB\n\n", mem_gb);
    fflush(stdout);

    /* ================================================================
     *  Step 3: Frobenius norm  ||A||_F^2 = sum |val|^2
     * ================================================================ */
    printf("[3/8] Frobenius norm ...\n"); fflush(stdout);
    double t = omp_get_wtime();

    double frob_sq = 0.0;
    #pragma omp parallel for num_threads(n_threads) reduction(+:frob_sq)
    for (int64_t e = 0; e < NNZ; e++) {
        float vr = crealf(vals[e]);
        float vi = cimagf(vals[e]);
        frob_sq += (double)vr * vr + (double)vi * vi;
    }

    printf("      ||A||_F  = %.6e\n", sqrt(frob_sq));
    printf("      ||A||_F^2= %.6e   (upper bound on ||A||^2)\n", frob_sq);
    printf("      Time: %.1f s\n\n", omp_get_wtime() - t);
    fflush(stdout);

    /* ================================================================
     *  Step 4: Estimate spectral norm ||A||^2 via power iteration
     *
     *  v_0 = random,  v_{i+1} = A^*A v_i / ||A^*A v_i||
     *  ||A||^2 ~ v_i^* A^*A v_i
     * ================================================================ */
    printf("[4/8] Power iteration for ||A||^2 (5 iterations) ...\n");
    fflush(stdout);
    t = omp_get_wtime();

    prng_t prng;
    prng.s[0] = 0x9E3779B97F4A7C15ULL;
    prng.s[1] = 0xC2B2AE3D27D4EB4FULL;

    /* Random start vector in R */
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t c = 0; c < N_COLS; c++)
        R[c] = prng_cnormal(&prng);

    /* Normalize */
    {
        double nrm_sq = 0.0;
        #pragma omp parallel for num_threads(n_threads) reduction(+:nrm_sq)
        for (int64_t c = 0; c < N_COLS; c++)
            nrm_sq += creal(conj(R[c]) * R[c]);
        dcomplex inv = 1.0 / sqrt(nrm_sq);
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int64_t c = 0; c < N_COLS; c++)
            R[c] *= inv;
    }

    double lambda_max = 0.0;
    for (int iter = 0; iter < 5; iter++) {
        /* Y = A @ R */
        matvec(row_idx, cols, vals, R, Y, n_threads);
        /* R = A^* @ Y  (this overwrites R, but that's fine) */
        rmatvec(row_idx, cols, vals, Y, R, local_bufs, n_threads);

        /* After these calls:
         *   Y = A v  (with ||v|| = 1)
         *   R = A^*A v
         * Rayleigh quotient:  lambda = v^* A^*A v = v^* R
         * But R has been overwritten with A^*A v, and v is gone.
         * Instead use:  ||A v||^2 = ||Y||^2 = v^* A^*A v  (since ||v||=1).
         */

        double av_sq = 0.0;
        #pragma omp parallel for num_threads(n_threads) reduction(+:av_sq)
        for (int64_t r = 0; r < N_ROWS; r++)
            av_sq += creal(conj(Y[r]) * Y[r]);

        lambda_max = av_sq;   /* ||A v||^2 with ||v||=1  =>  Rayleigh quotient */

        /* ||A^*A v||^2 for reporting */
        double rnorm_sq = 0.0;
        #pragma omp parallel for num_threads(n_threads) reduction(+:rnorm_sq)
        for (int64_t c = 0; c < N_COLS; c++)
            rnorm_sq += creal(conj(R[c]) * R[c]);

        /* Normalise R for next iteration */
        dcomplex inv = 1.0 / sqrt(rnorm_sq);
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int64_t c = 0; c < N_COLS; c++)
            R[c] *= inv;

        printf("      iter %d:  ||A v||^2 = %.6e   ||A^*A v|| = %.6e\n",
               iter, lambda_max, sqrt(rnorm_sq));
        fflush(stdout);
    }

    /* Spectral shift: c must be > ||A||^2 for convergence to null space */
    double c_shift = 1.05 * lambda_max;
    printf("      Estimated ||A||^2 = %.6e\n", lambda_max);
    printf("      Shift c = 1.05 * ||A||^2 = %.6e\n", c_shift);
    printf("      Time: %.1f s\n\n", omp_get_wtime() - t);
    fflush(stdout);

    /* ================================================================
     *  Step 5: Generate random Omega and orthonormalize -> Q
     * ================================================================ */
    printf("[5/8] Generating random subspace (k=%d) ...\n", k);
    fflush(stdout);
    t = omp_get_wtime();

    /* Seed PRNG (deterministic for reproducibility) */
    prng.s[0] = 0x123456789ABCDEF0ULL ^ ((uint64_t)k << 32) ^ (uint64_t)(iters);
    prng.s[1] = 0xFEDCBA9876543210ULL ^ ((uint64_t)k)       ^ (uint64_t)(iters + 1);

    for (int j = 0; j < k; j++) {
        dcomplex *col = Q + (int64_t)j * N_COLS;
        for (int64_t c = 0; c < N_COLS; c++)
            col[c] = prng_cnormal(&prng);
    }

    orthonormalize(Q, k);
    printf("      Time: %.1f s\n\n", omp_get_wtime() - t);
    fflush(stdout);

    /* ================================================================
     *  Step 6: Subspace iteration on B = cI - A^*A
     *
     *  for iter = 1 .. q:
     *    W = A^*A Q        (k matvecs + k rmatvecs)
     *    W = c*Q - W       (spectral shift)
     *    Q = orth(W)       (QR)
     * ================================================================ */
    printf("[6/8] Subspace iteration (%d iterations) ...\n", iters);
    fflush(stdout);
    t = omp_get_wtime();

    int ritz_check_interval = 5;   /* compute Ritz values every N iters */

    for (int iter = 0; iter < iters; iter++) {
        double ti = omp_get_wtime();

        /* W = A^*A Q  (column by column) */
        for (int j = 0; j < k; j++)
            apply_ata_column(row_idx, cols, vals, Q, W, Y, R,
                             local_bufs, n_threads, j);

        /* W = c*Q - W   (spectral shift: B = cI - A^*A) */
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int64_t i = 0; i < (int64_t)N_COLS * k; i++)
            W[i] = c_shift * Q[i] - W[i];

        /* Orthonormalize W, then swap so Q holds the result */
        orthonormalize(W, k);

        /* Swap Q and W */
        dcomplex *tmp = Q;  Q = W;  W = tmp;

        /* Periodic Ritz value check */
        if ((iter + 1) % ritz_check_interval == 0 || iter == iters - 1) {
            compute_ritz(row_idx, cols, vals, Q, W, G, eigenvalues,
                         Y, R, local_bufs, n_threads, k);

            printf("      iter %3d/%d  (%5.0fs):  smallest Ritz = %.3e  "
                   "largest = %.3e",
                   iter + 1, iters, omp_get_wtime() - ti,
                   eigenvalues[0], eigenvalues[k - 1]);
            /* Show the smallest few values */
            printf("  [");
            for (int i = 0; i < k && i < 8; i++) {
                printf("%.1e", eigenvalues[i]);
                if (i < k - 1 && i < 7) printf(" ");
            }
            if (k > 8) printf(" ...");
            printf("]\n");
            fflush(stdout);
        } else {
            printf("      iter %3d/%d  (%5.0fs)\n",
                   iter + 1, iters, omp_get_wtime() - ti);
            fflush(stdout);
        }
    }

    printf("      Total iteration time: %.0f s\n\n", omp_get_wtime() - t);
    fflush(stdout);

    /* ================================================================
     *  Step 7: Final Ritz values and nullity count
     * ================================================================ */
    printf("[7/8] Computing final Ritz values ...\n"); fflush(stdout);
    t = omp_get_wtime();

    compute_ritz(row_idx, cols, vals, Q, W, G, eigenvalues,
                 Y, R, local_bufs, n_threads, k);

    printf("      Time: %.0f s\n\n", omp_get_wtime() - t);
    printf("      All %d Ritz values of A^*A (ascending):\n", k);
    printf("      +---------------------------------------------+\n");
    for (int i = 0; i < k; i++) {
        printf("      |  lambda[%2d] = %+20.10e", i, eigenvalues[i]);
        /* Marker for near-zero values */
        double tol_val = pow(10.0, tol_exp) * eigenvalues[k - 1];
        if (eigenvalues[i] < tol_val)
            printf("  <<< near-zero");
        printf("\n");
    }
    printf("      +---------------------------------------------+\n\n");
    fflush(stdout);

    /* ---- Nullity detection ----
     *
     * Two methods:
     *   1. GAP-BASED (primary): find the largest log-gap between
     *      consecutive Ritz values.  The nullity is the number of
     *      values below the gap.
     *
     *   2. TOLERANCE-BASED (secondary): count Ritz values below
     *      10^(tol_exp) * lambda_max.
     *
     * The gap-based method is more robust because it doesn't require
     * the null space Ritz values to have fully converged to zero.
     */

    /* Method 1: Gap-based detection */
    int    best_gap_idx = -1;
    double best_gap     = 0.0;
    printf("\n      Gap analysis (ratios of consecutive eigenvalues):\n");
    for (int i = 1; i < k; i++) {
        if (eigenvalues[i - 1] > 0 && eigenvalues[i] > 0) {
            double ratio = eigenvalues[i] / eigenvalues[i - 1];
            double log_ratio = log10(ratio);
            printf("        lambda[%d]/lambda[%d] = %+.2e  (log10 = %+.2f)\n",
                   i, i - 1, ratio, log_ratio);
            if (log_ratio > best_gap) {
                best_gap = log_ratio;
                best_gap_idx = i;
            }
        }
    }
    int nullity_gap = (best_gap_idx >= 0) ? best_gap_idx : 0;
    if (best_gap_idx >= 0) {
        printf("\n      Largest gap between lambda[%d] and lambda[%d]"
               " (log10 = %.2f)\n",
               best_gap_idx - 1, best_gap_idx, best_gap);
        printf("      => Gap-based nullity estimate = %d\n", nullity_gap);
    }

    /* Method 2: Tolerance-based detection */
    double lambda_max_ritz = eigenvalues[k - 1];
    double tol = pow(10.0, tol_exp) * lambda_max_ritz;
    if (tol < 1e-300) tol = 1e-300;

    int nullity_tol = 0;
    for (int i = 0; i < k; i++) {
        if (eigenvalues[i] < tol)
            nullity_tol++;
    }
    printf("\n      Tolerance: %.6e  (10^(%.1f) * lambda_max = %.3e)\n",
           tol, tol_exp, lambda_max_ritz);
    printf("      => Tol-based nullity estimate = %d\n", nullity_tol);

    /* Use gap-based result as primary */
    int nullity = nullity_gap;

    /* ================================================================
     *  Step 8: Print final result
     * ================================================================ */
    printf("\n");
    printf("[8/8] RESULT\n\n");
    printf("    ============================================\n");
    printf("    ||  Estimated NULLITY = %-3d                ||\n", nullity);
    printf("    ||  Estimated RANK    = %-3lld               ||\n",
           (long long)(N_COLS - nullity));
    printf("    ============================================\n\n");

    if (nullity_tol != nullity_gap) {
        printf("    NOTE: Gap-based nullity (%d) differs from\n", nullity_gap);
        printf("          tol-based nullity (%d).  The gap-based\n", nullity_tol);
        printf("          estimate is more robust when null space\n");
        printf("          Ritz values have not fully converged.\n");
        printf("          Inspect the Ritz values above.\n\n");
    }

    /* ---- Summary table ---- */
    printf("    --------------------------------------------\n");
    printf("    Total wall time: %.0f s  (%.1f min)\n",
           omp_get_wtime() - T0, (omp_get_wtime() - T0) / 60.0);
    printf("    Matrix:      %lld x %lld,  NNZ = %lld\n",
           (long long)N_ROWS, (long long)N_COLS, (long long)NNZ);
    printf("    ||A||_F^2:   %.6e\n", frob_sq);
    printf("    ||A||^2:     %.6e  (estimated)\n", lambda_max);
    printf("    Shift c:     %.6e\n", c_shift);
    printf("    Subspace k:  %d\n", k);
    printf("    Iterations:  %d\n", iters);
    printf("    Threads:     %d\n", n_threads);
    printf("    --------------------------------------------\n");

    /* ---- Cleanup ---- */
    free(Q); free(W); free(Y); free(R);
    free(local_bufs); free(G); free(eigenvalues);

    munmap((void *)row_idx, sz_idx);
    munmap((void *)cols,    sz_cols);
    munmap((void *)vals,    sz_vals);

    return 0;
}
