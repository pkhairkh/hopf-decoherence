/*
 * verify_chain.c -- Verify ∂_b² ∘ ∂_b¹ = 0 (chain complex property)
 *
 * Takes random h ∈ C¹, computes v = ∂_b¹(h), then w = ∂_b²(v).
 * Checks ||w|| / ||v|| ≈ 0.
 *
 * Build: gcc -O3 -march=native -fopenmp -o verify_chain verify_chain.c -lm
 * Run:   ./verify_chain
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <omp.h>

typedef double _Complex dcomplex;
typedef float  _Complex scomplex;

#define TMPDIR "/tmp/h2b_sl3_box2ztl4"
#define N_ROWS_D2  1568987626LL
#define N_COLS      9448324LL
#define NNZ_D2     4091264798LL
#define N_COLS_D1   19522LL     /* C¹ shift (0,0) dimension */
#define D1_MAX_ROW  9448324LL   /* d1 row indices for shift (0,0) are < this */

static void *mmap_ro(const char *path, size_t sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    void *p = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, sz, MADV_RANDOM);
    return p;
}

/* PRNG */
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

int main(void) {
    int nt = omp_get_max_threads();
    double T0 = omp_get_wtime();

    printf("=== Verify ∂_b² ∘ ∂_b¹ = 0 ===\n\n");
    printf("Threads: %d\n\n", nt);
    fflush(stdout);

    /* ---- mmap d1 data ---- */
    printf("[1] mmap d1 data...\n"); fflush(stdout);
    struct stat st;
    stat(TMPDIR "/d1_row_idx.bin", &st);
    int64_t nnz_d1_total = st.st_size / 4;
    const int32_t *d1_row = mmap_ro(TMPDIR "/d1_row_idx.bin", nnz_d1_total * 4);
    const int32_t *d1_col = mmap_ro(TMPDIR "/d1_cols.bin", nnz_d1_total * 4);
    const scomplex *d1_val = mmap_ro(TMPDIR "/d1_vals.bin", nnz_d1_total * 8);
    printf("  d1 total nnz = %lld\n", (long long)nnz_d1_total);

    /* Count shift (0,0) entries */
    int64_t nnz_d1_s0 = 0;
    for (int64_t i = 0; i < nnz_d1_total; i++)
        if (d1_row[i] < D1_MAX_ROW) nnz_d1_s0++;
    printf("  d1 shift(0,0) nnz = %lld\n\n", (long long)nnz_d1_s0);
    fflush(stdout);

    /* ---- mmap d2 data (sorted) ---- */
    printf("[2] mmap d2 data (sorted)...\n"); fflush(stdout);
    const int32_t *d2_row = mmap_ro(TMPDIR "/row_idx_sorted.bin", NNZ_D2 * 4);
    const int32_t *d2_col = mmap_ro(TMPDIR "/cols_sorted.bin", NNZ_D2 * 4);
    const scomplex *d2_val = mmap_ro(TMPDIR "/vals_sorted.bin", NNZ_D2 * 8);
    printf("  d2 nnz = %lld\n\n", (long long)NNZ_D2);
    fflush(stdout);

    /* ---- Step 1: random h ∈ C¹, compute v = ∂_b¹(h) ---- */
    printf("[3] Compute v = d1 @ h (random h in C¹)...\n"); fflush(stdout);
    double t = omp_get_wtime();

    prng_t prng = {{42, 999}};
    dcomplex *h = calloc(N_COLS_D1, sizeof(dcomplex));
    dcomplex *v = calloc(N_COLS, sizeof(dcomplex));  /* v ∈ C² */

    for (int64_t i = 0; i < N_COLS_D1; i++)
        h[i] = prng_cn(&prng);

    /* v = ∂_b¹ @ h: for each d1 entry (row, col, val) with row < D1_MAX_ROW:
       v[row] += val * h[col]
       d1 is small (7.3M entries), use sequential loop. */
    for (int64_t i = 0; i < nnz_d1_total; i++) {
        if (d1_row[i] < D1_MAX_ROW) {
            int32_t r = d1_row[i];
            int32_t c = d1_col[i];
            if (c >= N_COLS_D1) {
                fprintf(stderr, "WARNING: d1_col[%lld] = %d >= %lld\n",
                        (long long)i, c, (long long)N_COLS_D1);
                continue;
            }
            v[r] += (dcomplex)d1_val[i] * h[c];
        }
    }

    double norm_v2 = 0;
    #pragma omp parallel for reduction(+:norm_v2)
    for (int64_t i = 0; i < N_COLS; i++)
        norm_v2 += creal(conj(v[i]) * v[i]);

    printf("  ||v|| = %.6e  (%.1fs)\n\n", sqrt(norm_v2), omp_get_wtime()-t);
    fflush(stdout);

    /* ---- Step 2: compute w = ∂_b²(v) ---- */
    printf("[4] Compute w = d2 @ v...\n"); fflush(stdout);
    t = omp_get_wtime();

    dcomplex *w = calloc(N_ROWS_D2, sizeof(dcomplex));  /* w ∈ C³ */
    /* d2 is sorted by row. Each thread handles a contiguous entry range. */
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk = NNZ_D2 / nthreads;
        int64_t e_start = (int64_t)tid * chunk;
        int64_t e_end = (tid == nthreads-1) ? NNZ_D2 : e_start + chunk;

        for (int64_t e = e_start; e < e_end; e++) {
            int32_t row = d2_row[e];
            int32_t col = d2_col[e];
            w[row] += (dcomplex)d2_val[e] * v[col];
        }
    }

    double norm_w2 = 0;
    #pragma omp parallel for reduction(+:norm_w2)
    for (int64_t i = 0; i < N_ROWS_D2; i++)
        norm_w2 += creal(conj(w[i]) * w[i]);

    printf("  ||w|| = %.6e  (%.1fs)\n\n", sqrt(norm_w2), omp_get_wtime()-t);
    fflush(stdout);

    /* ---- Step 3: check ratio ---- */
    printf("[5] Chain complex check:\n");
    printf("  ||d2(d1(h))|| / ||d1(h)|| = %.6e / %.6e = %.6e\n",
           sqrt(norm_w2), sqrt(norm_v2), sqrt(norm_w2) / sqrt(norm_v2));
    printf("  ||d2(d1(h))|| / ||h||     = %.6e / %.6e = %.6e\n",
           sqrt(norm_w2), sqrt(norm_v2), sqrt(norm_w2) / sqrt(norm_v2));

    double ratio = sqrt(norm_w2) / sqrt(norm_v2);
    if (ratio < 1e-6) {
        printf("\n  ✓ CHAIN COMPLEX PROPERTY HOLDS: d2 ∘ d1 ≈ 0\n");
        printf("  The augmented matrix construction is correct.\n");
    } else if (ratio < 1e-3) {
        printf("\n  ~ WEAK: d2 ∘ d1 is small but not zero (ratio = %.2e)\n", ratio);
        printf("  May be numerical precision issue (complex64 vals).\n");
    } else {
        printf("\n  ✗ CHAIN COMPLEX PROPERTY VIOLATED: d2 ∘ d1 ≠ 0\n");
        printf("  ratio = %.2e — the augmented matrix is WRONG.\n", ratio);
        printf("  Possible causes:\n");
        printf("    1. d1 and d2 use different indexing conventions\n");
        printf("    2. d2 formula (13 terms) has bugs\n");
        printf("    3. d1 and d2 operate on different spaces\n");
    }
    fflush(stdout);

    /* ---- Also test with a second random h ---- */
    printf("\n[6] Second test with different random h...\n"); fflush(stdout);
    prng.s[0] = 137; prng.s[1] = 42;
    for (int64_t i = 0; i < N_COLS_D1; i++) h[i] = prng_cn(&prng);
    memset(v, 0, N_COLS * sizeof(dcomplex));
    memset(w, 0, N_ROWS_D2 * sizeof(dcomplex));

    for (int64_t i = 0; i < nnz_d1_total; i++) {
        if (d1_row[i] < D1_MAX_ROW) {
            v[d1_row[i]] += (dcomplex)d1_val[i] * h[d1_col[i]];
        }
    }
    double nv2 = 0;
    for (int64_t i = 0; i < N_COLS; i++) nv2 += creal(conj(v[i])*v[i]);

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk = NNZ_D2 / nthreads;
        int64_t e_start = (int64_t)tid * chunk;
        int64_t e_end = (tid == nthreads-1) ? NNZ_D2 : e_start + chunk;
        for (int64_t e = e_start; e < e_end; e++)
            w[d2_row[e]] += (dcomplex)d2_val[e] * v[d2_col[e]];
    }
    double nw2 = 0;
    for (int64_t i = 0; i < N_ROWS_D2; i++) nw2 += creal(conj(w[i])*w[i]);

    printf("  ||v2|| = %.6e,  ||w2|| = %.6e,  ratio = %.6e\n",
           sqrt(nv2), sqrt(nw2), sqrt(nw2)/sqrt(nv2));
    fflush(stdout);

    /* ---- Cleanup ---- */
    free(h); free(v); free(w);
    printf("\nTotal time: %.1fs\n", omp_get_wtime()-T0);
    fflush(stdout);
    return 0;
}
