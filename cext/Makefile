# Makefile for pure C solvers (AMD EPYC-Turin, AVX-512, OpenMP)
CC = gcc
CFLAGS = -O3 -march=znver5 -mtune=znver5 -fopenmp -fno-math-errno -mcx16
LDFLAGS = -L/usr/lib64 -lopenblas -lgfortran -lm

BINS = sl3_projected_lanczos sl3_tracemin sl3_h2b verify_d2_d1_v2

all: $(BINS)

sl3_projected_lanczos: sl3_projected_lanczos.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sl3_tracemin: sl3_tracemin.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sl3_h2b: sl3_h2b.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

verify_d2_d1_v2: verify_d2_d1_v2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS)

.PHONY: all clean
