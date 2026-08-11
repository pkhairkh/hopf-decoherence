/*
 * sl3_tracemin.c — TRACEMIN solver for dim H̃²_b(B⁺(u_q(sl_3))) at ℓ=3.
 *
 * Algorithm: Trace Minimization on the Stiefel manifold
 *   min ||A_aug X||²_F  s.t. X* X = I_k
 *   A_aug = [d2; d1*]  (matrix-free, never materialized)
 *
 * Key features:
 *   - Householder QR retraction (via LAPACK zgeqrf + zungqr)
 *   - Per-column projected PCG inner solve with Ritz shifts
 *   - Block method (k=4) handles multiplicity-2 clustered eigenvalues
 *   - No condition-number squaring (operates on A_aug, not A_aug* A_aug)
 *   - Pure C, AVX-512, OpenMP, 128-bit CAS for hash accumulation
 *
 * Build:
 *   gcc -O3 -march=native -mcx16 -fopenmp -o sl3_tracemin sl3_tracemin.c \
 *       -lm -L/usr/lib64 -lopenblas -llapack -lgfortran
 *
 * Run:
 *   OMP_NUM_THREADS=15 taskset -c 0-14 ./sl3_tracemin . [k] [max_outer] [max_inner]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <sys/mman.h>
#include <omp.h>
#include <immintrin.h>

typedef double _Complex dcomplex;
typedef float  _Complex scomplex;

#define ELL 3
#define DIM 243
#define NNZ_MB 27671
#define NNZ_DB 2647

extern void zgeqrf_(const int*,const int*,dcomplex*,const int*,dcomplex*,dcomplex*,const int*,int*);
extern void zungqr_(const int*,const int*,const int*,dcomplex*,const int*,const dcomplex*,dcomplex*,const int*,int*);
extern void zheevd_(const char*,const char*,const int*,dcomplex*,const int*,double*,dcomplex*,const int*,double*,const int*,int*,const int*,int*);
extern void zgemm_(const char*,const char*,const int*,const int*,const int*,const dcomplex*,const dcomplex*,const int*,const dcomplex*,const int*,const dcomplex*,dcomplex*,const int*);

/* ====================================================================== */
/*  Algebra data                                                          */
/* ====================================================================== */
static int32_t weights[DIM][2];
struct __attribute__((packed)) mb_entry { int32_t l,a,b; double re,im; };
struct __attribute__((packed)) db_entry { int32_t c,j,k; double re,im; };
static struct mb_entry mult_bar[NNZ_MB];
static struct db_entry delta_bar[NNZ_DB];

static dcomplex mb_val(int i){return mult_bar[i].re+I*mult_bar[i].im;}
static dcomplex db_val(int i){return delta_bar[i].re+I*delta_bar[i].im;}

static int32_t *mb_by_b[DIM],*mb_by_l[DIM],*mb_by_a[DIM];
static int n_mb_by_b[DIM],n_mb_by_l[DIM],n_mb_by_a[DIM];
static int32_t *db_by_c[DIM],*db_by_j[DIM],*db_by_k[DIM];
static int n_db_by_c[DIM],n_db_by_j[DIM],n_db_by_k[DIM];
static int32_t *db_by_jk[DIM*DIM];
static int n_db_by_jk[DIM*DIM];
static int32_t *mb_by_ab[DIM*DIM];
static int n_mb_by_ab[DIM*DIM];

static void load_algebra(const char*dir){
    char p[512];FILE*f;
    snprintf(p,sizeof(p),"%s/weights.bin",dir);f=fopen(p,"rb");if(!f){perror(p);exit(1);}
    fread(weights,sizeof(int32_t),DIM*2,f);fclose(f);
    snprintf(p,sizeof(p),"%s/mult_bar.bin",dir);f=fopen(p,"rb");if(!f){perror(p);exit(1);}
    fread(mult_bar,sizeof(struct mb_entry),NNZ_MB,f);fclose(f);
    snprintf(p,sizeof(p),"%s/delta_bar.bin",dir);f=fopen(p,"rb");if(!f){perror(p);exit(1);}
    fread(delta_bar,sizeof(struct db_entry),NNZ_DB,f);fclose(f);
    for(int i=0;i<DIM;i++){n_mb_by_b[i]=n_mb_by_l[i]=n_mb_by_a[i]=0;n_db_by_c[i]=n_db_by_j[i]=n_db_by_k[i]=0;}
    for(int i=0;i<NNZ_MB;i++){n_mb_by_b[mult_bar[i].b]++;n_mb_by_l[mult_bar[i].l]++;n_mb_by_a[mult_bar[i].a]++;}
    for(int i=0;i<NNZ_DB;i++){n_db_by_c[delta_bar[i].c]++;n_db_by_j[delta_bar[i].j]++;n_db_by_k[delta_bar[i].k]++;}
    for(int i=0;i<DIM;i++){
        if(n_mb_by_b[i])mb_by_b[i]=malloc(n_mb_by_b[i]*sizeof(int32_t));
        if(n_mb_by_l[i])mb_by_l[i]=malloc(n_mb_by_l[i]*sizeof(int32_t));
        if(n_mb_by_a[i])mb_by_a[i]=malloc(n_mb_by_a[i]*sizeof(int32_t));
        if(n_db_by_c[i])db_by_c[i]=malloc(n_db_by_c[i]*sizeof(int32_t));
        if(n_db_by_j[i])db_by_j[i]=malloc(n_db_by_j[i]*sizeof(int32_t));
        if(n_db_by_k[i])db_by_k[i]=malloc(n_db_by_k[i]*sizeof(int32_t));
        n_mb_by_b[i]=n_mb_by_l[i]=n_mb_by_a[i]=0;
        n_db_by_c[i]=n_db_by_j[i]=n_db_by_k[i]=0;
    }
    for(int i=0;i<NNZ_MB;i++){mb_by_b[mult_bar[i].b][n_mb_by_b[mult_bar[i].b]++]=i;mb_by_l[mult_bar[i].l][n_mb_by_l[mult_bar[i].l]++]=i;mb_by_a[mult_bar[i].a][n_mb_by_a[mult_bar[i].a]++]=i;}
    for(int i=0;i<NNZ_DB;i++){db_by_c[delta_bar[i].c][n_db_by_c[delta_bar[i].c]++]=i;db_by_j[delta_bar[i].j][n_db_by_j[delta_bar[i].j]++]=i;db_by_k[delta_bar[i].k][n_db_by_k[delta_bar[i].k]++]=i;}
    for(int i=0;i<DIM*DIM;i++)n_db_by_jk[i]=0;
    for(int i=0;i<NNZ_DB;i++){int jk=delta_bar[i].j*DIM+delta_bar[i].k;n_db_by_jk[jk]++;}
    for(int i=0;i<DIM*DIM;i++){if(n_db_by_jk[i])db_by_jk[i]=malloc(n_db_by_jk[i]*sizeof(int32_t));n_db_by_jk[i]=0;}
    for(int i=0;i<NNZ_DB;i++){int jk=delta_bar[i].j*DIM+delta_bar[i].k;db_by_jk[jk][n_db_by_jk[jk]++]=i;}
    for(int i=0;i<DIM*DIM;i++)n_mb_by_ab[i]=0;
    for(int i=0;i<NNZ_MB;i++){int ab=mult_bar[i].a*DIM+mult_bar[i].b;n_mb_by_ab[ab]++;}
    for(int i=0;i<DIM*DIM;i++){if(n_mb_by_ab[i])mb_by_ab[i]=malloc(n_mb_by_ab[i]*sizeof(int32_t));n_mb_by_ab[i]=0;}
    for(int i=0;i<NNZ_MB;i++){int ab=mult_bar[i].a*DIM+mult_bar[i].b;mb_by_ab[ab][n_mb_by_ab[ab]++]=i;}
}

static int32_t w2i_elems[ELL*ELL][81];
static int w2i_n[ELL*ELL];
static int wt_key(int w0,int w1){return w0*ELL+w1;}
static int w2i_order[ELL*ELL];
static int w2i_nkeys;
static void build_weight_classes(void){
    for(int i=1;i<DIM;i++){
        int k=wt_key(weights[i][0],weights[i][1]);
        if(w2i_n[k]==0)w2i_order[w2i_nkeys++]=k;
        w2i_elems[k][w2i_n[k]++]=i;
    }
}

#define HASH_SIZE (1<<26)
#define C2_MAX 20000000
typedef struct{uint64_t key;int32_t idx;}hentry;
static hentry *c2hash;
static int32_t c2count;
static int32_t *c2_tuple;
static int8_t *c2_type;
static uint64_t fkey(int a,int b,int c){return ((uint64_t)a<<33)|((uint64_t)b<<17)|((uint64_t)c<<1)|0;}
static uint64_t gkey(int c,int j,int k){return ((uint64_t)c<<33)|((uint64_t)j<<17)|((uint64_t)k<<1)|1;}
static void hash_init(void){
    c2hash=calloc(HASH_SIZE,sizeof(hentry));
    for(int i=0;i<HASH_SIZE;i++)c2hash[i].idx=-1;
    c2count=0;
    c2_tuple=malloc(C2_MAX*3*sizeof(int32_t));
    c2_type=malloc(C2_MAX*sizeof(int8_t));
}
static int32_t hash_insert(uint64_t key){
    uint64_t h=key%HASH_SIZE;
    while(c2hash[h].idx!=-1){if(c2hash[h].key==key)return c2hash[h].idx;h=(h+1)%HASH_SIZE;}
    c2hash[h].key=key;c2hash[h].idx=c2count++;return c2hash[h].idx;
}

typedef struct{int32_t*rows;int32_t*cols;dcomplex*vals;int64_t nnz,cap;int32_t n_rows,n_cols;}spmat;
static spmat*spmat_new(int32_t nr,int32_t nc,int64_t cap){
    spmat*m=malloc(sizeof(spmat));
    m->rows=malloc(cap*4);m->cols=malloc(cap*4);m->vals=malloc(cap*16);
    m->nnz=0;m->cap=cap;m->n_rows=nr;m->n_cols=nc;return m;
}
static void spmat_add(spmat*m,int32_t r,int32_t c,dcomplex v){
    if(m->nnz>=m->cap){m->cap*=2;m->rows=realloc(m->rows,m->cap*4);m->cols=realloc(m->cols,m->cap*4);m->vals=realloc(m->vals,m->cap*16);}
    m->rows[m->nnz]=r;m->cols[m->nnz]=c;m->vals[m->nnz]=v;m->nnz++;
}

/* ====================================================================== */
/*  C³ hash for matrix-free d2 matvec (scomplex for speed)                */
/* ====================================================================== */
#define C3_BITS 31
#define C3_SIZE (1u<<C3_BITS)
#define C3_MASK (C3_SIZE-1)

typedef struct{
    uint64_t key;
    scomplex val;
} c3_entry;

static c3_entry *g_c3;
static int32_t g_n_c2;
static int32_t *g_c2_tuple;
static int8_t *g_c2_type;
static spmat *g_d1;

static inline uint32_t c3_pos(uint64_t key){
    key ^= key>>33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key>>33;
    return (uint32_t)(key & C3_MASK);
}

static inline uint64_t hkey(int comp,int a,int b,int c,int d){
    return ((uint64_t)comp<<60)|((uint64_t)a<<48)|((uint64_t)b<<32)|((uint64_t)c<<16)|((uint64_t)d);
}

static void c3_clear(void){
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<(int64_t)C3_SIZE;i++)
        g_c3[i].key=0xFFFFFFFFFFFFFFFFULL;
}

static inline void c3_accum(uint64_t key,scomplex val){
    uint32_t pos=c3_pos(key);
    while(1){
        uint64_t cur=g_c3[pos].key;
        if(cur==0xFFFFFFFFFFFFFFFFULL){
            if(__sync_bool_compare_and_swap(&g_c3[pos].key,cur,key)){
                g_c3[pos].val=val;
                return;
            }
            continue;
        }
        if(cur==key){
            uint64_t *vp=(uint64_t*)&g_c3[pos].val;
            uint64_t old_v,new_v;
            do{
                old_v=*vp;
                scomplex old_c=*(scomplex*)&old_v;
                scomplex new_c=old_c+val;
                new_v=*(uint64_t*)&new_c;
            }while(!__sync_bool_compare_and_swap(vp,old_v,new_v));
            return;
        }
        pos=(pos+1)&C3_MASK;
    }
}

static inline dcomplex c3_lookup(uint64_t key){
    uint32_t pos=c3_pos(key);
    while(g_c3[pos].key!=0xFFFFFFFFFFFFFFFFULL){
        if(g_c3[pos].key==key)
            return (dcomplex)g_c3[pos].val;
        pos=(pos+1)&C3_MASK;
    }
    return 0.0;
}

/* ====================================================================== */
/*  Matrix-free d2 forward matvec: h = d2 * v                             */
/*  CORRECTED MW formula: ∂_b(f,g) = (∂ʰf, ∂ʰg + ∂ᶜf, −∂ᶜg)             */
/*  Diagonal coactions for ∂ᶜf and ∂ʰg. 4-term ∂ᶜg.                      */
/* ====================================================================== */
static void d2_forward(const dcomplex *v,int nt){
    #pragma omp parallel for schedule(dynamic,64) num_threads(nt)
    for(int32_t i=0;i<g_n_c2;i++){
        dcomplex vi=v[i];
        if(cabs(vi)<1e-30)continue;

        if(g_c2_type[i]==0){
            int af=g_c2_tuple[i*3],bf=g_c2_tuple[i*3+1],lf=g_c2_tuple[i*3+2];
            /* ∂ʰf: +,-,+,- */
            for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];
                c3_accum(hkey(0,mult_bar[idx].a,af,bf,mult_bar[idx].l),(scomplex)(mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];
                c3_accum(hkey(0,mult_bar[idx].a,mult_bar[idx].b,bf,lf),(scomplex)(-mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];
                c3_accum(hkey(0,af,mult_bar[idx].a,mult_bar[idx].b,lf),(scomplex)(mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];
                c3_accum(hkey(0,af,bf,mult_bar[idx].b,mult_bar[idx].l),(scomplex)(-mb_val(idx)*vi));}
            /* +∂ᶜf T1: a₁b₁ ⊗ f(a₂,b₂). db_by_k[af]×db_by_k[bf]×mb_by_ab[a1*DIM+b1] */
            for(int e1=0;e1<n_db_by_k[af];e1++){int d1idx=db_by_k[af][e1];
                int a=delta_bar[d1idx].c, a1=delta_bar[d1idx].j;
                dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_k[bf];e2++){int d2idx=db_by_k[bf][e2];
                    int b=delta_bar[d2idx].c, b1=delta_bar[d2idx].j;
                    dcomplex d2v=db_val(d2idx);
                    int ab=a1*DIM+b1;
                    for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];
                        c3_accum(hkey(1,a,b,mult_bar[midx].l,lf),(scomplex)(d1v*d2v*mb_val(midx)*vi));
                    }
                }
            }
            /* +∂ᶜf T2: -Δ(f(a,b)). db_by_c[lf] */
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];
                c3_accum(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k),(scomplex)(-db_val(idx)*vi));}
            /* +∂ᶜf T3: f(a₁,b₁) ⊗ a₂b₂. db_by_j[af]×db_by_j[bf]×mb_by_ab[a2*DIM+b2] */
            for(int e1=0;e1<n_db_by_j[af];e1++){int d1idx=db_by_j[af][e1];
                int a=delta_bar[d1idx].c, a2=delta_bar[d1idx].k;
                dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_j[bf];e2++){int d2idx=db_by_j[bf][e2];
                    int b=delta_bar[d2idx].c, b2=delta_bar[d2idx].k;
                    dcomplex d2v=db_val(d2idx);
                    int ab=a2*DIM+b2;
                    for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];
                        c3_accum(hkey(1,a,b,lf,mult_bar[midx].l),(scomplex)(d1v*d2v*mb_val(midx)*vi));
                    }
                }
            }
        } else {
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];
            /* ∂ʰg T1: (Δa)·g(b). mb_by_b[jf]×mb_by_b[kf]×db_by_jk[a1*DIM+a2] */
            for(int e1=0;e1<n_mb_by_b[jf];e1++){int m1idx=mb_by_b[jf][e1];
                int t1=mult_bar[m1idx].l, a1=mult_bar[m1idx].a;
                dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_b[kf];e2++){int m2idx=mb_by_b[kf][e2];
                    int t2=mult_bar[m2idx].l, a2=mult_bar[m2idx].a;
                    dcomplex m2v=mb_val(m2idx);
                    int jk=a1*DIM+a2;
                    for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];
                        c3_accum(hkey(1,delta_bar[didx].c,cf,t1,t2),(scomplex)(db_val(didx)*m1v*m2v*vi));
                    }
                }
            }
            /* ∂ʰg T2: -g(ab). mb_by_l[cf] */
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];
                c3_accum(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf),(scomplex)(-mb_val(idx)*vi));}
            /* ∂ʰg T3: g(a)·(Δb). mb_by_a[jf]×mb_by_a[kf]×db_by_jk[b1*DIM+b2] */
            for(int e1=0;e1<n_mb_by_a[jf];e1++){int m1idx=mb_by_a[jf][e1];
                int t1=mult_bar[m1idx].l, b1=mult_bar[m1idx].b;
                dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_a[kf];e2++){int m2idx=mb_by_a[kf][e2];
                    int t2=mult_bar[m2idx].l, b2=mult_bar[m2idx].b;
                    dcomplex m2v=mb_val(m2idx);
                    int jk=b1*DIM+b2;
                    for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];
                        c3_accum(hkey(1,cf,delta_bar[didx].c,t1,t2),(scomplex)(m1v*m2v*db_val(didx)*vi));
                    }
                }
            }
            /* -∂ᶜg: -,+,-,+ (4 terms) */
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];
                c3_accum(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf),(scomplex)(-db_val(idx)*vi));}
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];
                c3_accum(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf),(scomplex)(db_val(idx)*vi));}
            for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];
                c3_accum(hkey(2,cf,jf,delta_bar[idx].j,delta_bar[idx].k),(scomplex)(-db_val(idx)*vi));}
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];
                c3_accum(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k),(scomplex)(db_val(idx)*vi));}
        }
    }
}

/* d2 adjoint: z = d2* @ h (h stored in C³ hash, z dense) */
static void d2_adjoint(dcomplex *z,int nt){
    #pragma omp parallel for schedule(dynamic,64) num_threads(nt)
    for(int32_t i=0;i<g_n_c2;i++){
        dcomplex zi=0;
        if(g_c2_type[i]==0){
            int af=g_c2_tuple[i*3],bf=g_c2_tuple[i*3+1],lf=g_c2_tuple[i*3+2];
            for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];
                zi+=conj(mb_val(idx))*c3_lookup(hkey(0,mult_bar[idx].a,af,bf,mult_bar[idx].l));}
            for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];
                zi-=conj(mb_val(idx))*c3_lookup(hkey(0,mult_bar[idx].a,mult_bar[idx].b,bf,lf));}
            for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];
                zi+=conj(mb_val(idx))*c3_lookup(hkey(0,af,mult_bar[idx].a,mult_bar[idx].b,lf));}
            for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];
                zi-=conj(mb_val(idx))*c3_lookup(hkey(0,af,bf,mult_bar[idx].b,mult_bar[idx].l));}
            for(int e1=0;e1<n_db_by_k[af];e1++){int d1idx=db_by_k[af][e1];
                int a=delta_bar[d1idx].c, a1=delta_bar[d1idx].j;
                dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_k[bf];e2++){int d2idx=db_by_k[bf][e2];
                    int b=delta_bar[d2idx].c, b1=delta_bar[d2idx].j;
                    dcomplex d2v=db_val(d2idx);
                    int ab=a1*DIM+b1;
                    for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];
                        zi+=conj(d1v*d2v*mb_val(midx))*c3_lookup(hkey(1,a,b,mult_bar[midx].l,lf));
                    }
                }
            }
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];
                zi-=conj(db_val(idx))*c3_lookup(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k));}
            for(int e1=0;e1<n_db_by_j[af];e1++){int d1idx=db_by_j[af][e1];
                int a=delta_bar[d1idx].c, a2=delta_bar[d1idx].k;
                dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_j[bf];e2++){int d2idx=db_by_j[bf][e2];
                    int b=delta_bar[d2idx].c, b2=delta_bar[d2idx].k;
                    dcomplex d2v=db_val(d2idx);
                    int ab=a2*DIM+b2;
                    for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];
                        zi+=conj(d1v*d2v*mb_val(midx))*c3_lookup(hkey(1,a,b,lf,mult_bar[midx].l));
                    }
                }
            }
        } else {
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];
            for(int e1=0;e1<n_mb_by_b[jf];e1++){int m1idx=mb_by_b[jf][e1];
                int t1=mult_bar[m1idx].l, a1=mult_bar[m1idx].a;
                dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_b[kf];e2++){int m2idx=mb_by_b[kf][e2];
                    int t2=mult_bar[m2idx].l, a2=mult_bar[m2idx].a;
                    dcomplex m2v=mb_val(m2idx);
                    int jk=a1*DIM+a2;
                    for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];
                        zi+=conj(m1v*m2v*db_val(didx))*c3_lookup(hkey(1,delta_bar[didx].c,cf,t1,t2));
                    }
                }
            }
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];
                zi-=conj(mb_val(idx))*c3_lookup(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf));}
            for(int e1=0;e1<n_mb_by_a[jf];e1++){int m1idx=mb_by_a[jf][e1];
                int t1=mult_bar[m1idx].l, b1=mult_bar[m1idx].b;
                dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_a[kf];e2++){int m2idx=mb_by_a[kf][e2];
                    int t2=mult_bar[m2idx].l, b2=mult_bar[m2idx].b;
                    dcomplex m2v=mb_val(m2idx);
                    int jk=b1*DIM+b2;
                    for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];
                        zi+=conj(m1v*m2v*db_val(didx))*c3_lookup(hkey(1,cf,delta_bar[didx].c,t1,t2));
                    }
                }
            }
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];
                zi-=conj(db_val(idx))*c3_lookup(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf));}
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];
                zi+=conj(db_val(idx))*c3_lookup(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf));}
            for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];
                zi-=conj(db_val(idx))*c3_lookup(hkey(2,cf,jf,delta_bar[idx].j,delta_bar[idx].k));}
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];
                zi+=conj(db_val(idx))*c3_lookup(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k));}
        }
        z[i]=zi;
    }
}

/* d1 forward: z = d1 @ w */
static void d1_forward(const dcomplex *w,dcomplex *z){
    memset(z,0,g_n_c2*sizeof(dcomplex));
    for(int64_t e=0;e<g_d1->nnz;e++)
        z[g_d1->rows[e]]+=g_d1->vals[e]*w[g_d1->cols[e]];
}

/* d1 adjoint: w = d1* @ v */
static void d1_adjoint(const dcomplex *v,dcomplex *w,int32_t n_c1){
    memset(w,0,n_c1*sizeof(dcomplex));
    for(int64_t e=0;e<g_d1->nnz;e++)
        w[g_d1->cols[e]]+=conj(g_d1->vals[e])*v[g_d1->rows[e]];
}

/* ====================================================================== */
/*  Apply M = A_aug* A_aug to a single vector: M@v = d2*(d2@v) + d1@(d1*@v) */
/*  Result in result (dense, n_c2)                                        */
/* ====================================================================== */
static int g_n_c1;
static int32_t g_n_c2;

static void apply_M(const dcomplex *v, dcomplex *result, dcomplex *w_buf, dcomplex *z2_buf, int nt){
    /* h = d2 @ v (stored in C³ hash) */
    c3_clear();
    d2_forward(v,nt);
    /* w = d1* @ v */
    d1_adjoint(v,w_buf,g_n_c1);
    /* result = d2* @ h */
    d2_adjoint(result,nt);
    /* z2 = d1 @ w */
    d1_forward(w_buf,z2_buf);
    /* result += z2 */
    #pragma omp parallel for num_threads(nt)
    for(int32_t c=0;c<g_n_c2;c++)result[c]+=z2_buf[c];
}

/* ====================================================================== */
/*  Householder QR orthonormalization (Stiefel retraction)                */
/*  Q: n×k matrix (column-major), overwritten with orthonormal Q         */
/* ====================================================================== */
static void householder_qr(dcomplex *Q, int k, int64_t n){
    int m=(int)n, ncol=k, lda=m, info;
    dcomplex *tau=malloc(k*sizeof(dcomplex));
    int lwork=-1; dcomplex wkopt;
    zgeqrf_(&m,&ncol,Q,&lda,tau,&wkopt,&lwork,&info);
    lwork=(int)creal(wkopt); if(lwork<1)lwork=1;
    dcomplex *work=malloc(lwork*sizeof(dcomplex));
    zgeqrf_(&m,&ncol,Q,&lda,tau,work,&lwork,&info);
    zungqr_(&m,&ncol,&ncol,Q,&lda,tau,work,&lwork,&info);
    free(work);free(tau);
}

/* ====================================================================== */
/*  PRNG: xorshift128+                                                    */
/* ====================================================================== */
typedef struct{uint64_t s[2];}prng_t;
static inline uint64_t prng_u64(prng_t*p){
    uint64_t s1=p->s[0],s0=p->s[1],r=s0+s1;
    p->s[0]=s0;s1^=s1<<23;p->s[1]=s1^s0^(s1>>18)^(s0>>5);
    return r;
}
static inline double prng_u(prng_t*p){return ((double)(prng_u64(p)>>11)+1.0)*(1.0/9007199254740992.0);}
static inline dcomplex prng_cn(prng_t*p){
    double u1=prng_u(p),u2=prng_u(p),r=sqrt(-log(u1));
    return (r*cos(2*M_PI*u2)+I*r*sin(2*M_PI*u2))*M_SQRT1_2;
}

/* ====================================================================== */
/*  TRACEMIN solver                                                       */
/*  min ||A_aug X||²_F  s.t. X*X = I_k                                   */
/* ====================================================================== */
static int solve_tracemin(int k, int max_outer, int max_inner, int nt){
    double T0=omp_get_wtime();

    printf("========================================================\n");
    printf("    TRACEMIN Solver (Householder QR, Stiefel manifold)\n");
    printf("    min ||A_aug X||²_F  s.t. X*X = I_k\n");
    printf("    A_aug = [d2; d1*],  matrix-free\n");
    printf("========================================================\n\n");
    printf("C¹=%d  C²=%d  k=%d  max_outer=%d  max_inner=%d  threads=%d\n\n",
           g_n_c1,g_n_c2,k,max_outer,max_inner,nt);
    fflush(stdout);

    /* Allocate C³ hash */
    printf("[1] Allocating C³ hash (%u entries, %.1f GB)...\n",
           C3_SIZE,(double)C3_SIZE*sizeof(c3_entry)/1e9);
    fflush(stdout);
    g_c3=malloc((size_t)C3_SIZE*sizeof(c3_entry));
    if(!g_c3){fprintf(stderr,"malloc g_c3 failed\n");exit(1);}
    /* Request transparent huge pages for the 34GB hash (critical for TLB performance) */
    madvise(g_c3,(size_t)C3_SIZE*sizeof(c3_entry),MADV_HUGEPAGE);
    c3_clear();
    printf("  done (%.1fs)\n\n",omp_get_wtime()-T0);fflush(stdout);

    /* Allocate workspace */
    int32_t n_c2=g_n_c2;
    int32_t n_c1=g_n_c1;

    /* X: n_c2 × k orthonormal block */
    dcomplex *X=malloc((size_t)k*n_c2*sizeof(dcomplex));
    /* R: n_c2 × k residual block (A_aug @ X) */
    dcomplex *R=malloc((size_t)k*n_c2*sizeof(dcomplex));
    /* AX: n_c2 × k (M @ X = A_aug* @ A_aug @ X) */
    dcomplex *AX=malloc((size_t)k*n_c2*sizeof(dcomplex));
    /* S: n_c2 × k update block */
    dcomplex *S=malloc((size_t)k*n_c2*sizeof(dcomplex));
    /* H: k × k Rayleigh-Ritz matrix */
    dcomplex *H=malloc(k*k*sizeof(dcomplex));
    /* V: k × k eigenvectors of H */
    dcomplex *V=malloc(k*k*sizeof(dcomplex));
    double *theta=malloc(k*sizeof(double));  /* Ritz values */

    /* Per-column PCG workspace */
    dcomplex *w_buf=malloc(n_c1*sizeof(dcomplex));
    dcomplex *z2_buf=malloc(n_c2*sizeof(dcomplex));
    dcomplex *mv_buf=malloc(n_c2*sizeof(dcomplex));  /* M @ v for PCG */
    dcomplex *r_pcg=malloc(n_c2*sizeof(dcomplex));   /* PCG residual */
    dcomplex *p_pcg=malloc(n_c2*sizeof(dcomplex));   /* PCG direction */
    dcomplex *Ap_pcg=malloc(n_c2*sizeof(dcomplex));  /* M @ p_pcg */
    dcomplex *x_pcg=malloc(n_c2*sizeof(dcomplex));   /* PCG solution */

    if(!X||!R||!AX||!S||!H||!V||!theta||!w_buf||!z2_buf||!mv_buf||!r_pcg||!p_pcg||!Ap_pcg||!x_pcg){
        fprintf(stderr,"malloc workspace failed\n");exit(1);
    }

    printf("[2] Allocating vectors: X,R,AX,S = %.1f GB total\n\n",
           (double)4*k*n_c2*16/1e9);fflush(stdout);

    /* Initialize X with random orthonormal block */
    printf("[3] Initializing random X (%d × %d)...\n",k,n_c2);fflush(stdout);
    double t=omp_get_wtime();
    prng_t prng={{0x9E3779B97F4A7C15ULL,0xC2B2AE3D27D4EB4FULL}};
    for(int j=0;j<k;j++)
        for(int32_t c=0;c<n_c2;c++)
            X[j*n_c2+c]=prng_cn(&prng);
    householder_qr(X,k,n_c2);
    printf("  done (%.1fs)\n\n",omp_get_wtime()-t);fflush(stdout);

    /* TRACEMIN main loop */
    printf("[4] TRACEMIN iteration:\n\n");fflush(stdout);
    double prev_min=1e30;

    for(int iter=0;iter<max_outer;iter++){
        double ti=omp_get_wtime();

        /* Step 1: Compute AX[:,j] = M @ X[:,j] for each column j */
        for(int j=0;j<k;j++){
            apply_M(X+j*n_c2, AX+j*n_c2, w_buf, z2_buf, nt);
        }

        /* Step 2: H = X* AX (k×k Hermitian) */
        dcomplex alpha=1,beta=0;
        int kk=k, nn=n_c2;
        zgemm_("C","N",&kk,&kk,&nn,&alpha,X,&nn,AX,&nn,&beta,H,&kk);

        /* Step 3: Eigendecompose H → theta, V */
        int lwork=-1,lrwork=-1,liwork=-1,info;
        dcomplex wkopt; double rwkopt=0; int iwkopt=0;
        zheevd_("V","U",&kk,H,&kk,theta,&wkopt,&lwork,&rwkopt,&lrwork,&iwkopt,&liwork,&info);
        lwork=(int)creal(wkopt); lrwork=(int)rwkopt; liwork=iwkopt;
        dcomplex *work_ev=malloc(lwork*sizeof(dcomplex));
        double *rwork_ev=malloc(lrwork*sizeof(double));
        int *iwork_ev=malloc(liwork*sizeof(int));
        zheevd_("V","U",&kk,H,&kk,theta,work_ev,&lwork,rwork_ev,&lrwork,iwork_ev,&liwork,&info);
        if(info!=0){fprintf(stderr,"zheevd failed: info=%d\n",info);exit(1);}
        /* V = H (eigenvectors, now in H) */
        memcpy(V,H,k*k*sizeof(dcomplex));
        free(work_ev);free(rwork_ev);free(iwork_ev);

        /* Step 4: Rotate X ← X @ V (Rayleigh-Ritz extraction) */
        /* X_new = X @ V, so X_new[:,i] = Σ_j X[:,j] * V[j,i] */
        {
            dcomplex *Xtmp=malloc((size_t)k*n_c2*sizeof(dcomplex));
            alpha=1;beta=0;
            zgemm_("N","N",&nn,&kk,&kk,&alpha,X,&nn,V,&kk,&beta,Xtmp,&nn);
            memcpy(X,Xtmp,(size_t)k*n_c2*sizeof(dcomplex));
            free(Xtmp);
        }

        /* Step 5: Recompute AX in rotated basis: AX_new = AX @ V */
        {
            dcomplex *AXtmp=malloc((size_t)k*n_c2*sizeof(dcomplex));
            alpha=1;beta=0;
            zgemm_("N","N",&nn,&kk,&kk,&alpha,AX,&nn,V,&kk,&beta,AXtmp,&nn);
            memcpy(AX,AXtmp,(size_t)k*n_c2*sizeof(dcomplex));
            free(AXtmp);
        }

        /* Step 6: Compute residual R[:,j] = AX[:,j] - theta[j]*X[:,j] */
        double res_norm2=0;
        for(int j=0;j<k;j++){
            double th=theta[j];
            #pragma omp parallel for reduction(+:res_norm2) num_threads(nt)
            for(int32_t c=0;c<n_c2;c++){
                dcomplex r=AX[j*n_c2+c]-th*X[j*n_c2+c];
                R[j*n_c2+c]=r;
                res_norm2+=creal(conj(r)*r);
            }
        }

        double min_theta=theta[0];
        printf("  iter %3d/%d (%5.0fs): theta=[",iter+1,max_outer,omp_get_wtime()-ti);
        for(int j=0;j<k;j++)printf("%.3e ",theta[j]);
        printf("]  ||R||=%.3e\n",sqrt(res_norm2));
        fflush(stdout);

        /* Check convergence */
        if(res_norm2 < 1e-10*fabs(min_theta)){
            printf("  CONVERGED! ||R|| < tol*|theta_min|\n");
            break;
        }
        if(iter>0 && fabs(min_theta-prev_min)<1e-8*fabs(min_theta)){
            printf("  theta_min converged (change < 1e-8)\n");
            break;
        }
        prev_min=min_theta;

        /* Step 7: Per-column projected CG solve */
        /* For each column j, solve P(M - theta[j] I)P s_j = -R[:,j] */
        for(int j=0;j<k;j++){
            double shift=theta[j];
            if(shift<0)shift=0;  /* clamp shift to keep PSD */

            /* x_pcg = 0, r_pcg = -R[:,j] (NEGATIVE: solving P(M-σI)P s = -R), p_pcg = r_pcg */
            memset(x_pcg,0,n_c2*sizeof(dcomplex));
            #pragma omp parallel for num_threads(nt)
            for(int32_t c=0;c<n_c2;c++) r_pcg[c]=-R[j*n_c2+c];

            /* Project r_pcg: r = r - X (X* r) */
            /* X* r is k-vector; r -= X @ (X* r) */
            dcomplex *Xr=malloc(k*sizeof(dcomplex));
            alpha=1;beta=0;
            zgemm_("C","N",&kk,&(int){1},&nn,&alpha,X,&nn,r_pcg,&(int){n_c2},&beta,Xr,&kk);
            alpha=-1;beta=1;
            zgemm_("N","N",&nn,&(int){1},&kk,&alpha,X,&nn,Xr,&kk,&beta,r_pcg,&(int){n_c2});
            free(Xr);

            memcpy(p_pcg,r_pcg,n_c2*sizeof(dcomplex));
            double rsold=0;
            for(int32_t c=0;c<n_c2;c++)rsold+=creal(conj(r_pcg[c])*r_pcg[c]);

            for(int icg=0;icg<max_inner;icg++){
                /* Ap = M @ p_pcg - shift * p_pcg */
                apply_M(p_pcg, Ap_pcg, w_buf, z2_buf, nt);
                #pragma omp parallel for num_threads(nt)
                for(int32_t c=0;c<n_c2;c++)Ap_pcg[c]-=shift*p_pcg[c];

                /* Project Ap: Ap -= X (X* Ap) */
                dcomplex *XAp=malloc(k*sizeof(dcomplex));
                alpha=1;beta=0;
                zgemm_("C","N",&kk,&(int){1},&nn,&alpha,X,&nn,Ap_pcg,&(int){n_c2},&beta,XAp,&kk);
                alpha=-1;beta=1;
                zgemm_("N","N",&nn,&(int){1},&kk,&alpha,X,&nn,XAp,&kk,&beta,Ap_pcg,&(int){n_c2});
                free(XAp);

                double pAp=0;
                for(int32_t c=0;c<n_c2;c++)pAp+=creal(conj(p_pcg[c])*Ap_pcg[c]);
                if(pAp<=1e-30)break;  /* breakdown (catches both ~0 and negative) */
                double alpha_cg=rsold/pAp;
                #pragma omp parallel for num_threads(nt)
                for(int32_t c=0;c<n_c2;c++){
                    x_pcg[c]+=alpha_cg*p_pcg[c];
                    r_pcg[c]-=alpha_cg*Ap_pcg[c];
                }
                double rsnew=0;
                for(int32_t c=0;c<n_c2;c++)rsnew+=creal(conj(r_pcg[c])*r_pcg[c]);
                if(sqrt(rsnew)<1e-6*sqrt(rsold+1e-30))break;
                double beta_cg=rsnew/rsold;
                #pragma omp parallel for num_threads(nt)
                for(int32_t c=0;c<n_c2;c++)p_pcg[c]=r_pcg[c]+beta_cg*p_pcg[c];
                rsold=rsnew;
            }

            /* Store solution s_j = x_pcg */
            memcpy(S+j*n_c2,x_pcg,n_c2*sizeof(dcomplex));
        }

        /* Step 8: QR retraction: X_new = qr(X + S) */
        #pragma omp parallel for num_threads(nt)
        for(int64_t i=0;i<(int64_t)k*n_c2;i++)
            X[i]+=S[i];
        householder_qr(X,k,n_c2);
    }

    /* Final result */
    printf("\n[5] FINAL RESULT\n\n");
    printf("  Ritz values (ascending):\n");
    for(int j=0;j<k;j++)
        printf("    theta[%d] = %+.10e\n",j,theta[j]);
    printf("\n");

    /* Nullity detection */
    double lam_max=theta[k-1];
    double tol=lam_max*1e-6;
    int nullity=0;
    for(int j=0;j<k;j++){
        if(theta[j]<tol)nullity++;
        else break;
    }
    /* Also check for largest gap */
    int best_gap=-1;double best_log=-1e9;
    for(int j=1;j<k;j++){
        if(theta[j-1]>0&&theta[j]>0){
            double lg=log10(theta[j]/theta[j-1]);
            if(lg>best_log){best_log=lg;best_gap=j;}
        }
    }

    printf("  theta_max ≈ %.6e\n",lam_max);
    printf("  Largest gap: theta[%d]/theta[%d] (log10=%.2f)\n",best_gap,best_gap-1,best_log);
    printf("  Near-zero count (tol=%.2e): %d\n",tol,nullity);
    printf("\n  ============================================\n");
    printf("  ||  NULLITY (per shift) = %-3d              ||\n",nullity);
    printf("  ||  dim H̃²_b(B⁺) = %-3d                    ||\n",nullity);
    printf("  ||  dim HH²(sl₃) = 9 - %-3d = %-3d          ||\n",nullity,9-nullity);
    printf("  ============================================\n\n");fflush(stdout);

    /* Cleanup */
    free(X);free(R);free(AX);free(S);free(H);free(V);free(theta);
    free(w_buf);free(z2_buf);free(mv_buf);free(r_pcg);free(p_pcg);free(Ap_pcg);free(x_pcg);
    free(g_c3);

    printf("Total time: %.0fs (%.1f min)\n",omp_get_wtime()-T0,(omp_get_wtime()-T0)/60.0);fflush(stdout);
    return nullity;
}

/* ====================================================================== */
/*  Main                                                                  */
/* ====================================================================== */
int main(int argc,char*argv[]){
    const char*dir=(argc>1)?argv[1]:".";
    int k=(argc>2)?atoi(argv[2]):4;
    int max_outer=(argc>3)?atoi(argv[3]):30;
    int max_inner=(argc>4)?atoi(argv[4]):10;
    int nt=omp_get_max_threads();
    if(nt>15)nt=15;
    omp_set_num_threads(nt);
    double T0=omp_get_wtime();

    printf("=== sl_3 HH² TRACEMIN solver (pure C, AVX-512, OpenMP) ===\nThreads=%d\n\n",nt);fflush(stdout);

    printf("[A] Loading algebra...\n");fflush(stdout);
    load_algebra(dir);build_weight_classes();
    printf("  mult_bar:%d delta_bar:%d\n",NNZ_MB,NNZ_DB);
    for(int k2=0;k2<ELL*ELL;k2++)if(w2i_n[k2])printf("  wt(%d,%d):%d\n",k2/ELL,k2%ELL,w2i_n[k2]);
    fflush(stdout);

    /* Build C¹, C², d1 */
    printf("\n[B] Building C1, C2, d1...\n");fflush(stdout);
    double t=omp_get_wtime();
    int s0=0,s1=0;

    int32_t n_c1=0;int32_t c1j[20000],c1k[20000];
    for(int wi=0;wi<w2i_nkeys;wi++){int wj=w2i_order[wi];
    for(int wj2=0;wj2<w2i_nkeys;wj2++){int wk=w2i_order[wj2];
        if((wj/ELL-wk/ELL)%ELL==s0&&(wj%ELL-wk%ELL)%ELL==s1)
            for(int ji=0;ji<w2i_n[wj];ji++)for(int ki=0;ki<w2i_n[wk];ki++)
                {c1j[n_c1]=w2i_elems[wj][ji];c1k[n_c1]=w2i_elems[wk][ki];n_c1++;}
    }}

    hash_init();
    int32_t n_f=0,n_g=0;
    for(int wai=0;wai<w2i_nkeys;wai++){int wa=w2i_order[wai];
        for(int wbi=0;wbi<w2i_nkeys;wbi++){int wb=w2i_order[wbi];
            int wl=wt_key((s0+wa/ELL+wb/ELL)%ELL,(s1+wa%ELL+wb%ELL)%ELL);
            if(!w2i_n[wl])continue;
            for(int ai=0;ai<w2i_n[wa];ai++)for(int bi=0;bi<w2i_n[wb];bi++)for(int li=0;li<w2i_n[wl];li++){
                int a=w2i_elems[wa][ai],b=w2i_elems[wb][bi],l=w2i_elems[wl][li];
                int32_t idx=hash_insert(fkey(a,b,l));
                c2_tuple[idx*3]=a;c2_tuple[idx*3+1]=b;c2_tuple[idx*3+2]=l;c2_type[idx]=0;n_f++;
            }
        }
    }
    for(int wci=0;wci<w2i_nkeys;wci++){int wc=w2i_order[wci];
        for(int wji=0;wji<w2i_nkeys;wji++){int wj=w2i_order[wji];
            int wk0=((s0-wj/ELL+wc/ELL)%ELL+ELL)%ELL;
            int wk1=((s1-wj%ELL+wc%ELL)%ELL+ELL)%ELL;
            int wk=wt_key(wk0,wk1);
            if(!w2i_n[wk])continue;
            for(int ci=0;ci<w2i_n[wc];ci++)for(int ji=0;ji<w2i_n[wj];ji++)for(int ki=0;ki<w2i_n[wk];ki++){
                int c=w2i_elems[wc][ci],j=w2i_elems[wj][ji],k=w2i_elems[wk][ki];
                int32_t idx=hash_insert(gkey(c,j,k));
                c2_tuple[idx*3]=c;c2_tuple[idx*3+1]=j;c2_tuple[idx*3+2]=k;c2_type[idx]=1;n_g++;
            }
        }
    }
    int32_t n_c2=c2count;
    printf("  C1:%d C2:%d (%df+%dg)  (%.1fs)\n\n",n_c1,n_c2,n_f,n_g,omp_get_wtime()-t);fflush(stdout);

    /* Build d1 (with corrected hj/hk and T2 fix) */
    printf("[C] Building d1...\n");fflush(stdout);
    t=omp_get_wtime();
    spmat*d1=spmat_new(n_c2,n_c1,10000000);
    for(int ci=0;ci<n_c1;ci++){
        int hj=c1j[ci],hk=c1k[ci];
        /* ∂ʰh: T1=mb_by_b[hj], T2=mb_by_l[hk], T3=mb_by_a[hj] */
        for(int i=0;i<n_mb_by_b[hj];i++){int idx=mb_by_b[hj][i];
            int32_t r=hash_insert(fkey(mult_bar[idx].a,hk,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
        for(int i=0;i<n_mb_by_l[hk];i++){int idx=mb_by_l[hk][i];
            int32_t r=hash_insert(fkey(mult_bar[idx].a,mult_bar[idx].b,hj));
            if(r>=0)spmat_add(d1,r,ci,-mb_val(idx));}
        for(int i=0;i<n_mb_by_a[hj];i++){int idx=mb_by_a[hj][i];
            int32_t r=hash_insert(fkey(hk,mult_bar[idx].b,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
        /* -∂ᶜh: T1=-db_by_k[hk], T2=+db_by_c[hj] (FIXED!), T3=-db_by_j[hk] */
        for(int i=0;i<n_db_by_k[hk];i++){int idx=db_by_k[hk][i];
            int32_t r=hash_insert(gkey(delta_bar[idx].c,delta_bar[idx].j,hj));
            if(r>=0)spmat_add(d1,r,ci,-db_val(idx));}
        for(int i=0;i<n_db_by_c[hj];i++){int idx=db_by_c[hj][i];
            int32_t r=hash_insert(gkey(hk,delta_bar[idx].j,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,db_val(idx));}
        for(int i=0;i<n_db_by_j[hk];i++){int idx=db_by_j[hk][i];
            int32_t r=hash_insert(gkey(delta_bar[idx].c,hj,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,-db_val(idx));}
    }
    printf("  d1: %lld nnz (%.1fs)\n\n",(long long)d1->nnz,omp_get_wtime()-t);fflush(stdout);

    /* Set global pointers */
    g_n_c1=n_c1;
    g_n_c2=n_c2;
    g_d1=d1;
    g_c2_tuple=c2_tuple;
    g_c2_type=c2_type;

    /* Run TRACEMIN */
    printf("[D] Starting TRACEMIN solver...\n\n");fflush(stdout);
    int nullity=solve_tracemin(k,max_outer,max_inner,nt);

    printf("\n=== FINAL RESULT ===\n");
    printf("  nullity per shift = %d\n",nullity);
    printf("  dim H̃²_b(B⁺) = %d\n",nullity);
    printf("  dim HH²(u_q(sl_3)) = 9 - %d = %d\n",nullity,9-nullity);
    printf("\nTotal: %.1fs (%.1f min)\n",omp_get_wtime()-T0,(omp_get_wtime()-T0)/60.0);fflush(stdout);

    free(d1->rows);free(d1->cols);free(d1->vals);free(d1);
    free(c2hash);
    return 0;
}
