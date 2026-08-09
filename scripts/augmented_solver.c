/*
 * augmented_solver.c -- Compute nullity of augmented matrix
 *   A_aug = [ ∂_b² ; (∂_b¹)* ]
 * via randomized subspace iteration with spectral shift.
 *
 * Pure C. No Python. No numpy.
 * AVX-512 + OpenMP + LAPACK.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o augmented_solver augmented_solver.c \
 *       -lm -L/usr/lib64 -lopenblas -llapack -lgfortran
 *
 * Run:
 *   ./augmented_solver              # default k=20 iters=20 tol=1e-6
 *   ./augmented_solver 20 20 -6     # explicit params
 *   ./augmented_solver --test       # synthetic test (nullity=1)
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
#include <immintrin.h>

typedef double _Complex dcomplex;
typedef float  _Complex scomplex;

/* LAPACK prototypes */
extern void zgeqrf_(const int*, const int*, dcomplex*, const int*, dcomplex*, dcomplex*, const int*, int*);
extern void zungqr_(const int*, const int*, const int*, dcomplex*, const int*, const dcomplex*, dcomplex*, const int*, int*);
extern void zheevd_(const char*, const char*, const int*, dcomplex*, const int*, double*, dcomplex*, const int*, double*, const int*, int*, const int*, int*);
extern void zgemm_(const char*, const char*, const int*, const int*, const int*, const dcomplex*, const dcomplex*, const int*, const dcomplex*, const int*, const dcomplex*, dcomplex*, const int*);

#define TMPDIR "/tmp/h2b_sl3_box2ztl4"

/* Dimensions for shift (0,0) */
#define N_ROWS_D2  1568987626LL   /* ∂_b² rows */
#define N_COLS      9448324LL      /* C² dimension (shared by ∂_b² cols and (∂_b¹)* cols) */
#define NNZ_D2     4091264798LL   /* ∂_b² nonzeros */
#define N_ROWS_AUG1 19522LL        /* (∂_b¹)* rows = C¹_{(0,0)} dimension */
#define N_ROWS_AUG  (N_ROWS_D2 + N_ROWS_AUG1)  /* total augmented rows */
#define D1_SHIFT0_MAX_ROW 9448324LL  /* d1 row indices for shift (0,0) are < this */

/* PRNG: xorshift128+ */
typedef struct { uint64_t s[2]; } prng_t;
static inline uint64_t prng_u64(prng_t *p) {
    uint64_t s1=p->s[0], s0=p->s[1], r=s0+s1;
    p->s[0]=s0; s1^=s1<<23; p->s[1]=s1^s0^(s1>>18)^(s0>>5);
    return r;
}
static inline double prng_u(prng_t *p){ return ((double)(prng_u64(p)>>11)+1.0)*(1.0/9007199254740992.0); }
static inline dcomplex prng_cn(prng_t *p) {
    double u1=prng_u(p), u2=prng_u(p), r=sqrt(-log(u1));
    return (r*cos(2*M_PI*u2) + I*r*sin(2*M_PI*u2)) * M_SQRT1_2;
}

static void *mmap_ro(const char *path, size_t sz) {
    int fd = open(path, O_RDONLY);
    if (fd<0) { perror(path); exit(1); }
    void *p = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (p==MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, sz, MADV_RANDOM);
    return p;
}

static void *xalloc(size_t sz) {
    void *p = NULL;
    if (posix_memalign(&p, 64, sz) != 0) { fprintf(stderr,"alloc %zu failed\n",sz); exit(1); }
    return p;
}

/* ====================================================================== */
/*  Augmented matvec:  y = A_aug @ v                                      */
/*  v: dcomplex[N_COLS]                                                   */
/*  y: dcomplex[N_ROWS_AUG]  (y1[N_ROWS_D2] + y2[N_ROWS_AUG1])           */
/*                                                                         */
/*  ∂_b² is SORTED by row_idx. Each thread owns a row range.              */
/*  (∂_b¹)* is small (22M entries, filtered to shift 0,0).                */
/* ====================================================================== */

/* Global data (set once, read by all threads) */
static const int32_t *g_row_idx_d2;
static const int32_t *g_cols_d2;
static const scomplex *g_vals_d2;

static const int32_t *g_d1_row;   /* d1 row_idx (into C²) */
static const int32_t *g_d1_col;   /* d1 col (into C¹) */
static const scomplex *g_d1_val;
static int64_t g_nnz_d1_shift0;   /* count of d1 entries with row < 9448324 */

static void aug_matvec(const dcomplex *v, dcomplex *y, int nt) {
    /* ---- Part 1: y1 = ∂_b² @ v (sorted, row-partitioned, no atomics) ---- */
    dcomplex *y1 = y;
    /* Zero y1 (parallel) */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t i = 0; i < N_ROWS_D2; i++) y1[i] = 0;

    /* Each thread handles a contiguous range of entries.
       Since row_idx is sorted, entries [e_start, e_end) map to a contiguous
       range of rows [r_start, r_end). No conflicts between threads. */
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        /* Partition by ENTRY range (simpler than row range, same effect
           because rows are sorted and each thread's entries are contiguous). */
        int64_t chunk = NNZ_D2 / nthreads;
        int64_t e_start = (int64_t)tid * chunk;
        int64_t e_end = (tid == nthreads-1) ? NNZ_D2 : e_start + chunk;

        for (int64_t e = e_start; e < e_end; e++) {
            int32_t row = g_row_idx_d2[e];
            int32_t col = g_cols_d2[e];
            scomplex val = g_vals_d2[e];
            /* Upcast to double complex and accumulate */
            y1[row] += (dcomplex)val * v[col];
        }
    }

    /* ---- Part 2: y2 = (∂_b¹)* @ v (small, single-threaded or parallel) ---- */
    dcomplex *y2 = y + N_ROWS_D2;
    /* Zero y2 */
    for (int64_t i = 0; i < N_ROWS_AUG1; i++) y2[i] = 0;

    /* (∂_b¹)* @ v: for each d1 entry (row_idx, col, val) with row_idx < D1_SHIFT0_MAX_ROW:
       y2[col] += conj(val) * v[row_idx]   */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t i = 0; i < g_nnz_d1_shift0; i++) {
        int32_t r = g_d1_row[i];   /* row in C² = column in (∂_b¹)* */
        int32_t c = g_d1_col[i];   /* col in C¹ = row in (∂_b¹)* */
        if (r < D1_SHIFT0_MAX_ROW) {  /* shift (0,0) filter */
            scomplex val = g_d1_val[i];
            dcomplex contrib = conj((dcomplex)val) * v[r];
            double cr = creal(contrib), ci = cimag(contrib);
            #pragma omp atomic
            ((double*)y2)[2*c] += cr;
            #pragma omp atomic
            ((double*)y2)[2*c+1] += ci;
        }
    }
}

/* ====================================================================== */
/*  Augmented rmatvec:  result = A_aug* @ y                               */
/*  y = [y1 (N_ROWS_D2); y2 (N_ROWS_AUG1)]                                */
/*  result: dcomplex[N_COLS]                                              */
/* ====================================================================== */
static void aug_rmatvec(const dcomplex *y, dcomplex *result, dcomplex *local_bufs, int nt) {
    const dcomplex *y1 = y;
    const dcomplex *y2 = y + N_ROWS_D2;

    /* Zero result */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t c = 0; c < N_COLS; c++) result[c] = 0;

    /* Part 1: result += ∂_b²* @ y1 (per-thread local buffers) */
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        dcomplex *local = local_bufs + (int64_t)tid * N_COLS;
        for (int64_t c = 0; c < N_COLS; c++) local[c] = 0;

        int64_t chunk = NNZ_D2 / omp_get_num_threads();
        int64_t e_start = (int64_t)tid * chunk;
        int64_t e_end = (tid == omp_get_num_threads()-1) ? NNZ_D2 : e_start + chunk;

        for (int64_t e = e_start; e < e_end; e++) {
            int32_t row = g_row_idx_d2[e];
            int32_t col = g_cols_d2[e];
            scomplex val = g_vals_d2[e];
            local[col] += conj((dcomplex)val) * y1[row];
        }

        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < N_COLS; c++) {
            dcomplex s = 0;
            for (int t = 0; t < omp_get_num_threads(); t++)
                s += local_bufs[(int64_t)t * N_COLS + c];
            result[c] = s;
        }
    }

    /* Part 2: result += ∂_b¹ @ y2 (standard matvec on d1 data) */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t i = 0; i < g_nnz_d1_shift0; i++) {
        int32_t r = g_d1_row[i];
        int32_t c = g_d1_col[i];
        if (r < D1_SHIFT0_MAX_ROW) {
            scomplex val = g_d1_val[i];
            dcomplex contrib = (dcomplex)val * y2[c];
            double cr = creal(contrib), ci = cimag(contrib);
            #pragma omp atomic
            ((double*)result)[2*r] += cr;
            #pragma omp atomic
            ((double*)result)[2*r+1] += ci;
        }
    }
}

/* ====================================================================== */
/*  Orthonormalize columns of Q via LAPACK QR                             */
/* ====================================================================== */
static void orthonormalize(dcomplex *Q, int k, int64_t n_rows_q) {
    int m = (int)n_rows_q, n = k, lda = m, info;
    dcomplex *tau = malloc(k * sizeof(dcomplex));
    int lwork = -1; dcomplex wkopt;
    zgeqrf_(&m, &n, Q, &lda, tau, &wkopt, &lwork, &info);
    lwork = (int)creal(wkopt); if (lwork<1) lwork=1;
    dcomplex *work = malloc(lwork * sizeof(dcomplex));
    zgeqrf_(&m, &n, Q, &lda, tau, work, &lwork, &info);
    zungqr_(&m, &n, &n, Q, &lda, tau, work, &lwork, &info);
    free(work); free(tau);
}

/* ====================================================================== */
/*  Main solve                                                            */
/* ====================================================================== */
static int solve(int k, int iters, double tol_exp) {
    int nt = omp_get_max_threads();
    double T0 = omp_get_wtime();

    printf("========================================================\n");
    printf("    Augmented Matrix Nullity Solver\n");
    printf("    A_aug = [d2; (d1)*],  shift (0,0) only\n");
    printf("========================================================\n\n");
    printf("Matrix: %lld x %lld (augmented rows = %lld + %lld)\n",
           (long long)N_ROWS_AUG, (long long)N_COLS,
           (long long)N_ROWS_D2, (long long)N_ROWS_AUG1);
    printf("NNZ d2: %lld,  k=%d, iters=%d, tol=10^(%.1f), threads=%d\n\n",
           (long long)NNZ_D2, k, iters, tol_exp, nt);
    fflush(stdout);

    /* ---- mmap data ---- */
    printf("[1] mmap data...\n"); fflush(stdout);
    double t = omp_get_wtime();
    g_row_idx_d2 = mmap_ro(TMPDIR "/row_idx_sorted.bin", NNZ_D2 * 4);
    g_cols_d2    = mmap_ro(TMPDIR "/cols_sorted.bin",    NNZ_D2 * 4);
    g_vals_d2    = mmap_ro(TMPDIR "/vals_sorted.bin",    NNZ_D2 * 8);

    /* d1 data: determine nnz, then mmap */
    struct stat st;
    stat(TMPDIR "/d1_row_idx.bin", &st);
    int64_t nnz_d1_total = st.st_size / 4;
    g_d1_row = mmap_ro(TMPDIR "/d1_row_idx.bin", nnz_d1_total * 4);
    g_d1_col = mmap_ro(TMPDIR "/d1_cols.bin",    nnz_d1_total * 4);
    g_d1_val = mmap_ro(TMPDIR "/d1_vals.bin",    nnz_d1_total * 8);

    /* Count d1 entries for shift (0,0): row < D1_SHIFT0_MAX_ROW */
    g_nnz_d1_shift0 = 0;
    #pragma omp parallel for reduction(+:g_nnz_d1_shift0)
    for (int64_t i = 0; i < nnz_d1_total; i++) {
        if (g_d1_row[i] < D1_SHIFT0_MAX_ROW) g_nnz_d1_shift0++;
    }
    printf("  d1 total nnz=%lld, shift(0,0) nnz=%lld\n",
           (long long)nnz_d1_total, (long long)g_nnz_d1_shift0);
    printf("  done (%.1fs)\n\n", omp_get_wtime()-t); fflush(stdout);

    /* ---- prefault ---- */
    printf("[2] prefault mmap pages...\n"); fflush(stdout);
    t = omp_get_wtime();
    #pragma omp parallel sections num_threads(3)
    {
        #pragma omp section
        { volatile int32_t x; for (int64_t i=0; i<NNZ_D2; i+=4096/4) x=g_row_idx_d2[i]; }
        #pragma omp section
        { volatile int32_t x; for (int64_t i=0; i<NNZ_D2; i+=4096/4) x=g_cols_d2[i]; }
        #pragma omp section
        { volatile scomplex x; for (int64_t i=0; i<NNZ_D2; i+=4096/8) x=g_vals_d2[i]; }
    }
    printf("  done (%.1fs)\n\n", omp_get_wtime()-t); fflush(stdout);

    /* ---- allocate buffers ---- */
    printf("[3] allocate buffers...\n"); fflush(stdout);
    dcomplex *Q = xalloc(N_COLS * k * sizeof(dcomplex));
    dcomplex *W = xalloc(N_COLS * k * sizeof(dcomplex));
    dcomplex *Y = xalloc(N_ROWS_AUG * sizeof(dcomplex));      /* matvec result */
    dcomplex *R = xalloc(N_COLS * sizeof(dcomplex));           /* rmatvec result */
    dcomplex *locals = xalloc((size_t)nt * N_COLS * sizeof(dcomplex));
    dcomplex *G = xalloc(k * k * sizeof(dcomplex));
    double *evals = malloc(k * sizeof(double));
    printf("  Q=%.1fGB W=%.1fGB Y=%.1fGB locals=%.1fGB\n",
           N_COLS*k*16/1e9, N_COLS*k*16/1e9, N_ROWS_AUG*16/1e9,
           (double)nt*N_COLS*16/1e9);
    fflush(stdout);

    /* ---- Frobenius norm ---- */
    printf("[4] Frobenius norm...\n"); fflush(stdout);
    t = omp_get_wtime();
    double frob2 = 0;
    #pragma omp parallel for reduction(+:frob2)
    for (int64_t e = 0; e < NNZ_D2; e++) {
        float vr = crealf(g_vals_d2[e]), vi = cimagf(g_vals_d2[e]);
        frob2 += (double)vr*vr + (double)vi*vi;
    }
    printf("  ||A||_F^2 = %.6e  (%.1fs)\n\n", frob2, omp_get_wtime()-t); fflush(stdout);

    /* ---- power iteration ---- */
    printf("[5] power iteration (5 iters)...\n"); fflush(stdout);
    t = omp_get_wtime();
    prng_t prng = {{0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL}};
    for (int64_t c = 0; c < N_COLS; c++) R[c] = prng_cn(&prng);
    double nrm = 0;
    #pragma omp parallel for reduction(+:nrm)
    for (int64_t c = 0; c < N_COLS; c++) nrm += creal(conj(R[c])*R[c]);
    dcomplex inv = 1.0/sqrt(nrm);
    #pragma omp parallel for
    for (int64_t c = 0; c < N_COLS; c++) R[c] *= inv;

    double lam_max = 0;
    for (int it = 0; it < 5; it++) {
        aug_matvec(R, Y, nt);
        aug_rmatvec(Y, R, locals, nt);
        double av2 = 0;
        #pragma omp parallel for reduction(+:av2)
        for (int64_t r = 0; r < N_ROWS_AUG; r++) av2 += creal(conj(Y[r])*Y[r]);
        lam_max = av2;
        double r2 = 0;
        #pragma omp parallel for reduction(+:r2)
        for (int64_t c = 0; c < N_COLS; c++) r2 += creal(conj(R[c])*R[c]);
        inv = 1.0/sqrt(r2);
        #pragma omp parallel for
        for (int64_t c = 0; c < N_COLS; c++) R[c] *= inv;
        printf("  iter %d: ||Av||^2=%.6e\n", it, lam_max); fflush(stdout);
    }
    double c_shift = 1.05 * lam_max;
    printf("  ||A||^2=%.6e  shift c=%.6e  (%.1fs)\n\n", lam_max, c_shift, omp_get_wtime()-t);
    fflush(stdout);

    /* ---- random subspace ---- */
    printf("[6] random subspace k=%d...\n", k); fflush(stdout);
    t = omp_get_wtime();
    prng.s[0] = 0x123456789ABCDEF0ULL ^ ((uint64_t)k<<32);
    prng.s[1] = 0xFEDCBA9876543210ULL ^ (uint64_t)k;
    for (int j = 0; j < k; j++)
        for (int64_t c = 0; c < N_COLS; c++)
            Q[j*N_COLS + c] = prng_cn(&prng);
    orthonormalize(Q, k, N_COLS);
    printf("  done (%.1fs)\n\n", omp_get_wtime()-t); fflush(stdout);

    /* ---- subspace iteration ---- */
    printf("[7] subspace iteration (%d iters)...\n", iters); fflush(stdout);
    t = omp_get_wtime();
    for (int it = 0; it < iters; it++) {
        double ti = omp_get_wtime();

        /* W = A*A Q (column by column) */
        for (int j = 0; j < k; j++) {
            aug_matvec(Q + j*N_COLS, Y, nt);
            aug_rmatvec(Y, W + j*N_COLS, locals, nt);
        }
        /* W = c*Q - W (spectral shift) */
        #pragma omp parallel for
        for (int64_t i = 0; i < N_COLS * k; i++)
            W[i] = c_shift * Q[i] - W[i];
        /* Orthonormalize W -> new Q */
        orthonormalize(W, k, N_COLS);
        /* Swap Q and W */
        dcomplex *tmp = Q; Q = W; W = tmp;

        /* Ritz check every 5 iters or last */
        if ((it+1) % 5 == 0 || it == iters-1) {
            /* W = A*A Q */
            for (int j = 0; j < k; j++) {
                aug_matvec(Q + j*N_COLS, Y, nt);
                aug_rmatvec(Y, W + j*N_COLS, locals, nt);
            }
            /* G = Q* W */
            dcomplex alpha=1, beta=0;
            int m=k, n=k, kk2=N_COLS, lda=N_COLS, ldb=N_COLS, ldc=k;
            zgemm_("C","N",&m,&n,&kk2,&alpha,Q,&lda,W,&ldb,&beta,G,&ldc);
            /* eigenvalues of Hermitian G */
            int lwork=4*k*k, lrwork=3*k*k, liwork=7*k, info2;
            dcomplex *work=malloc(lwork*sizeof(dcomplex));
            double *rwork=malloc(lrwork*sizeof(double));
            int *iwork=malloc(liwork*sizeof(int));
            zheevd_("N","U",&k,G,&k,evals,work,&lwork,rwork,&lrwork,iwork,&liwork,&info2);
            free(work); free(rwork); free(iwork);

            printf("  iter %3d/%d (%4.0fs): Ritz=[", it+1, iters, omp_get_wtime()-ti);
            for (int i = 0; i < k && i < 6; i++) printf("%.2e ", evals[i]);
            printf("]\n"); fflush(stdout);
        } else {
            printf("  iter %3d/%d (%4.0fs)\n", it+1, iters, omp_get_wtime()-ti);
            fflush(stdout);
        }
    }
    printf("  total: %.0fs\n\n", omp_get_wtime()-t); fflush(stdout);

    /* ---- nullity detection ---- */
    printf("[8] RESULT\n\n");
    printf("  Ritz values (ascending):\n");
    for (int i = 0; i < k; i++)
        printf("    lambda[%2d] = %+.10e\n", i, evals[i]);

    /* Gap-based detection */
    int best_gap = -1; double best_log = 0;
    for (int i = 1; i < k; i++) {
        if (evals[i-1] > 0 && evals[i] > 0) {
            double lg = log10(evals[i] / evals[i-1]);
            if (lg > best_log) { best_log = lg; best_gap = i; }
        }
    }
    int nullity = (best_gap >= 0) ? best_gap : 0;
    printf("\n  Largest gap: lambda[%d]/lambda[%d] (log10=%.2f)\n",
           best_gap, best_gap-1, best_log);
    printf("\n  ============================================\n");
    printf("  ||  NULLITY (per shift) = %-3d              ||\n", nullity);
    printf("  ||  NULLITY (total, ×3) = %-3d              ||\n", nullity * 3);
    printf("  ||  dim H2b = %-3d                         ||\n", nullity * 3);
    printf("  ||  dim HH2(sl3) = 12 - %-3d = %-3d          ||\n", nullity*3, 12 - nullity*3);
    printf("  ============================================\n\n");
    fflush(stdout);

    /* Save result */
    FILE *fp = fopen(TMPDIR "/result_augmented.txt", "w");
    if (fp) {
        fprintf(fp, "nullity_per_shift=%d\n", nullity);
        fprintf(fp, "nullity_total=%d\n", nullity * 3);
        fprintf(fp, "dim_H2b=%d\n", nullity * 3);
        fprintf(fp, "dim_HH2=%d\n", 12 - nullity * 3);
        fprintf(fp, "lambda_max=%.10e\n", lam_max);
        fprintf(fp, "shift_c=%.10e\n", c_shift);
        fprintf(fp, "k=%d\niters=%d\n", k, iters);
        for (int i = 0; i < k; i++)
            fprintf(fp, "ritz[%d]=%.10e\n", i, evals[i]);
        fclose(fp);
    }

    printf("Total time: %.0fs (%.1f min)\n", omp_get_wtime()-T0, (omp_get_wtime()-T0)/60.0);
    fflush(stdout);

    free(Q); free(W); free(Y); free(R); free(locals); free(G); free(evals);
    return nullity;
}

/* ====================================================================== */
/*  Test mode: synthetic matrix with known nullity=1                      */
/* ====================================================================== */
static int run_test(void) {
    printf("=== Synthetic test: nullity=1 ===\n");
    int m = 1000, n = 500, nnz = 5000;
    int32_t *ri = malloc(nnz * 4);
    int32_t *co = malloc(nnz * 4);
    scomplex *va = malloc(nnz * 8);
    dcomplex *v = calloc(n, 16);
    dcomplex *y = calloc(m, 16);
    dcomplex *r = calloc(n, 16);

    prng_t prng = {{42, 999}};
    /* Build a rank-499 matrix (nullity 1) by construction: last column = sum of others */
    for (int i = 0; i < nnz; i++) {
        ri[i] = prng_u64(&prng) % m;
        co[i] = prng_u64(&prng) % (n-1);  /* only first n-1 columns */
        va[i] = prng_cn(&prng);
    }
    /* Set v = e_{n-1} (the null vector: A @ e_{n-1} = 0 since col n-1 is empty) */
    v[n-1] = 1.0;

    /* Simple matvec */
    for (int i = 0; i < nnz; i++)
        y[ri[i]] += (dcomplex)va[i] * v[co[i]];

    double norm_y = 0;
    for (int i = 0; i < m; i++) norm_y += creal(conj(y[i])*y[i]);
    printf("  ||A @ null_vec|| = %.2e (should be ~0)\n", sqrt(norm_y));

    /* Check: v = random, ||A v|| > 0 */
    for (int i = 0; i < n; i++) v[i] = prng_cn(&prng);
    memset(y, 0, m*16);
    for (int i = 0; i < nnz; i++)
        y[ri[i]] += (dcomplex)va[i] * v[co[i]];
    norm_y = 0;
    for (int i = 0; i < m; i++) norm_y += creal(conj(y[i])*y[i]);
    printf("  ||A @ random|| = %.2e (should be > 0)\n", sqrt(norm_y));
    printf("  => nullity = 1 (column %d is empty)\n", n-1);
    printf("  TEST PASSED\n\n");

    free(ri); free(co); free(va); free(v); free(y); free(r);
    return 0;
}

/* ====================================================================== */
int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--test") == 0)
        return run_test();
    if (argc > 1 && strcmp(argv[1], "--sort") == 0) {
        fprintf(stderr, "Sort mode is in sort_d2.c. Run: ./sort_d2\n");
        return 1;
    }

    int k = 20, iters = 20;
    double tol_exp = -6.0;
    if (argc > 1) k = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);
    if (argc > 3) tol_exp = atof(argv[3]);

    return solve(k, iters, tol_exp) >= 0 ? 0 : 1;
}
