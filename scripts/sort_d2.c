/*
 * sort_d2.c -- Parallel counting sort of ∂_b² entries by row_idx.
 * Pure C, OpenMP, no Python/numpy.
 *
 * Build: gcc -O3 -fopenmp -o sort_d2 sort_d2.c -lm
 * Run:   ./sort_d2
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
#include <time.h>
#include <omp.h>

#define NNZ      4091264798LL
#define N_ROWS   1568987626LL
#define TMPDIR   "/tmp/h2b_sl3_box2ztl4"

static double now(void) { return omp_get_wtime(); }

static void *mmap_ro(const char *path, size_t sz)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    void *p = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, sz, MADV_SEQUENTIAL);
    return p;
}

static void *mmap_rw(const char *path, size_t sz)
{
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror(path); exit(1); }
    if (ftruncate(fd, sz) < 0) { perror("ftruncate"); exit(1); }
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap rw"); exit(1); }
    madvise(p, sz, MADV_SEQUENTIAL);
    return p;
}

int main(void)
{
    int nt = omp_get_max_threads();
    double t0 = now();
    printf("=== Counting sort of d2 by row_idx ===\n");
    printf("NNZ=%lld N_ROWS=%lld threads=%d\n", (long long)NNZ, (long long)N_ROWS, nt);
    fflush(stdout);

    /* ---- mmap inputs ---- */
    size_t sz_idx  = (size_t)NNZ * 4;
    size_t sz_cols = (size_t)NNZ * 4;
    size_t sz_vals = (size_t)NNZ * 8;

    printf("[1] mmap inputs...\n"); fflush(stdout);
    const int32_t *row_idx = mmap_ro(TMPDIR "/row_idx_flat.bin", sz_idx);
    const int32_t *cols    = mmap_ro(TMPDIR "/cols_flat.bin",    sz_cols);
    const float _Complex *vals = mmap_ro(TMPDIR "/vals_flat.bin", sz_vals);
    printf("    done (%.1fs)\n", now()-t0); fflush(stdout);

    /* ---- histogram ---- */
    printf("[2] histogram (atomic, %d threads)...\n", nt); fflush(stdout);
    double t1 = now();
    int64_t *hist = calloc(N_ROWS + 1, sizeof(int64_t));
    if (!hist) { fprintf(stderr, "calloc hist failed\n"); exit(1); }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < NNZ; i++) {
        int32_t r = row_idx[i];
        #pragma omp atomic
        hist[r]++;
    }
    printf("    done (%.1fs), total=%lld\n", now()-t1, (long long)hist[N_ROWS]);
    fflush(stdout);

    /* ---- prefix sum (exclusive) ---- */
    printf("[3] prefix sum...\n"); fflush(stdout);
    double t2 = now();
    int64_t sum = 0;
    for (int64_t i = 0; i <= N_ROWS; i++) {
        int64_t tmp = hist[i];
        hist[i] = sum;
        sum += tmp;
    }
    printf("    done (%.1fs), last=%lld (expect %lld)\n", now()-t2, (long long)hist[N_ROWS], (long long)NNZ);
    fflush(stdout);

    /* ---- create + mmap outputs ---- */
    printf("[4] create output files...\n"); fflush(stdout);
    double t3 = now();
    int32_t *out_idx  = mmap_rw(TMPDIR "/row_idx_sorted.bin", sz_idx);
    int32_t *out_cols = mmap_rw(TMPDIR "/cols_sorted.bin",    sz_cols);
    float _Complex *out_vals = mmap_rw(TMPDIR "/vals_sorted.bin", sz_vals);
    printf("    done (%.1fs)\n", now()-t3); fflush(stdout);

    /* ---- scatter ---- */
    printf("[5] scatter (atomic capture, %d threads)...\n", nt); fflush(stdout);
    double t4 = now();
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < NNZ; i++) {
        int32_t r = row_idx[i];
        int64_t pos;
        #pragma omp atomic capture
        pos = hist[r]++;
        out_idx[pos]  = r;
        out_cols[pos] = cols[i];
        out_vals[pos] = vals[i];
    }
    printf("    done (%.1fs)\n", now()-t4); fflush(stdout);

    /* ---- verify ---- */
    printf("[6] verify...\n"); fflush(stdout);
    int ok = 1;
    for (int64_t i = 1; i < 1000 && i < NNZ; i++) {
        if (out_idx[i] < out_idx[i-1]) { ok = 0; break; }
    }
    int64_t tail = NNZ - 10;
    for (int64_t i = tail + 1; i < NNZ; i++) {
        if (out_idx[i] < out_idx[i-1]) { ok = 0; break; }
    }
    printf("    first 5: %d %d %d %d %d\n", out_idx[0], out_idx[1], out_idx[2], out_idx[3], out_idx[4]);
    printf("    last 5:  %d %d %d %d %d\n", out_idx[NNZ-5], out_idx[NNZ-4], out_idx[NNZ-3], out_idx[NNZ-2], out_idx[NNZ-1]);
    printf("    %s\n", ok ? "SORT OK" : "SORT FAILED");
    fflush(stdout);

    /* ---- cleanup ---- */
    munmap((void*)row_idx, sz_idx);
    munmap((void*)cols, sz_cols);
    munmap((void*)vals, sz_vals);
    munmap(out_idx, sz_idx);
    munmap(out_cols, sz_cols);
    munmap(out_vals, sz_vals);
    free(hist);

    printf("\n=== SORT COMPLETE === (%.1fs)\n", now()-t0);
    fflush(stdout);
    return 0;
}
