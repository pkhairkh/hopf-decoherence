/*
 * sl3_solve.c — Complete computation of dim HH²(u_q(sl_3)) at ℓ=3.
 *
 * Pure C. No Python. No numpy. No disk I/O for d2.
 *
 * Strategy:
 *   1. Load algebra, build C¹, C², d1 (from sl3_h2b.c)
 *   2. Matrix-free augmented solver:
 *      - A_aug = [d2; d1*] (conceptual, never materialized)
 *      - d2*v computed on-the-fly via 10-term formula + C³ hash
 *      - Subspace iteration to find nullity(A_aug*A_aug) = dim H̃²_b
 *   3. dim HH² = 12 - 3 × dim H̃²_b
 *
 * d2 formula (CORRECTED signs):
 *   ∂_b(f,g) = (∂ʰf, ∂ʰg + ∂ᶜf, -∂ᶜg)
 *   ∂ʰf (4 terms, signs +,-,+,-)  → h-rows
 *   ∂ᶜf (3 terms, signs -,+,−)    → m-rows  [p=2, (-1)^(p+1),(-1)^p,(-1)^(p+q+1)]
 *   ∂ʰg (3 terms, signs +,-,+)    → m-rows
 *   ∂ᶜg (4 terms, signs +,-,+,-)  → c-rows  [p=1, 2 coactions + 2 Δ-on-outputs]
 *   -∂ᶜg (4 terms, signs -,+,-,+) → c-rows
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o sl3_solve sl3_solve.c \
 *       -lm -L/usr/lib64 -lopenblas -llapack -lgfortran
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
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
}

/* ====================================================================== */
/*  Weight classes                                                        */
/* ====================================================================== */
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

/* ====================================================================== */
/*  C² hash (f-rows and g-rows in ONE table)                             */
/* ====================================================================== */
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

/* ====================================================================== */
/*  d1 sparse matrix                                                      */
/* ====================================================================== */
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
static void spmat_free(spmat*m){free(m->rows);free(m->cols);free(m->vals);free(m);}

/* ====================================================================== */
/*  C³ hash for matrix-free d2 matvec                                     */
/*  Key encoding: comp(4 bits, <<60) | a(16 bits, <<48) | b(16, <<32)    */
/*                | c(16, <<16) | d(16, <<0)                              */
/*  Sentinel: key = 0xFFFFFFFFFFFFFFFF                                    */
/* ====================================================================== */
#define C3_GLOBAL_BITS 31
#define C3_GLOBAL_SIZE (1u<<C3_GLOBAL_BITS)
#define C3_GLOBAL_MASK (C3_GLOBAL_SIZE-1)

#define C3_LOCAL_BITS  27
#define C3_LOCAL_SIZE  (1u<<C3_LOCAL_BITS)
#define C3_LOCAL_MASK  (C3_LOCAL_SIZE-1)

typedef struct{
    uint64_t key;
    scomplex val;  /* complex64 to save memory */
} c3_entry;

static c3_entry *g_c3_global;  /* global C³ hash, 2^31 × 16B = 34GB */

/* Hash function: mix bits for uniform distribution */
static inline uint32_t c3_hash_pos(uint64_t key){
    key ^= key>>33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key>>33;
    return (uint32_t)(key & C3_GLOBAL_MASK);
}

/* C³ row key encoders */
static inline uint64_t hkey(int comp,int a,int b,int c,int d){
    return ((uint64_t)comp<<60)|((uint64_t)a<<48)|((uint64_t)b<<32)|((uint64_t)c<<16)|((uint64_t)d);
}

static void c3_global_clear(void){
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<(int64_t)C3_GLOBAL_SIZE;i++)
        g_c3_global[i].key=0xFFFFFFFFFFFFFFFFULL;
}

/* Atomic accumulate into global hash (thread-safe via CAS) */
static inline void c3_global_accumulate(uint64_t key,scomplex val){
    uint32_t pos=c3_hash_pos(key);
    while(1){
        uint64_t cur_key=g_c3_global[pos].key;
        if(cur_key==0xFFFFFFFFFFFFFFFFULL){
            if(__sync_bool_compare_and_swap(&g_c3_global[pos].key,cur_key,key)){
                g_c3_global[pos].val=val;
                return;
            }
            continue;
        }
        if(cur_key==key){
            /* Atomic add for complex64 via CAS loop on uint64 representation */
            uint64_t *vp=(uint64_t*)&g_c3_global[pos].val;
            uint64_t old_v,new_v;
            do{
                old_v=*vp;
                scomplex old_c=*(scomplex*)&old_v;
                scomplex new_c=old_c+val;
                new_v=*(uint64_t*)&new_c;
            }while(!__sync_bool_compare_and_swap(vp,old_v,new_v));
            return;
        }
        pos=(pos+1)&C3_GLOBAL_MASK;
    }
}

/* Lookup in global hash, return complex128 value or 0 (read-only) */
static inline dcomplex c3_global_lookup(uint64_t key){
    uint32_t pos=c3_hash_pos(key);
    while(g_c3_global[pos].key!=0xFFFFFFFFFFFFFFFFULL){
        if(g_c3_global[pos].key==key)
            return (dcomplex)g_c3_global[pos].val;
        pos=(pos+1)&C3_GLOBAL_MASK;
    }
    return 0.0;
}

/* ====================================================================== */
/*  Matrix-free d2 forward matvec: h = d2 * v                             */
/*  v: dcomplex[n_c2] (dense)                                             */
/*  Result stored in g_c3_global hash (key → scomplex value)             */
/*                                                                        */
/*  For each C² entry i with value v[i]:                                 */
/*    f-type (af,bf,lf):                                                  */
/*      ∂ʰf T1: +mb_val(t,a,lf) → hkey(0,a,af,bf,t)                      */
/*      ∂ʰf T2: -mb_val(af,a,b) → hkey(0,a,b,bf,lf)                      */
/*      ∂ʰf T3: +mb_val(bf,b,c) → hkey(0,af,b,c,lf)                      */
/*      ∂ʰf T4: -mb_val(t,lf,c) → hkey(0,af,bf,c,t)                      */
/*      +∂ᶜf T1: -db_val(a,j,af) → hkey(1,a,bf,j,lf)                     */
/*      +∂ᶜf T2: +db_val(lf,j,k) → hkey(1,af,bf,j,k)                     */
/*      +∂ᶜf T3: -db_val(b,bf,k) → hkey(1,af,b,lf,k)                     */
/*    g-type (cf,jf,kf):                                                  */
/*      ∂ʰg T1: +mb_val(j,a,jf) → hkey(1,a,cf,j,kf)                      */
/*      ∂ʰg T2: -mb_val(cf,a,b) → hkey(1,a,b,jf,kf)                      */
/*      ∂ʰg T3: +mb_val(k,b,kf) → hkey(1,cf,b,jf,k)                      */
/*      -∂ᶜg T1: -db_val(c,j,cf) → hkey(2,c,j,jf,kf)                     */
/*      -∂ᶜg T2: +db_val(jf,j,K) → hkey(2,cf,j,K,kf)                     */
/*      -∂ᶜg T3: -db_val(kf,j,K) → hkey(2,cf,jf,j,K)                     */
/*      -∂ᶜg T4: +db_val(c,cf,l) → hkey(2,c,jf,kf,l)                     */
/* ====================================================================== */
static int32_t g_n_c2;
static int32_t *g_c2_tuple;
static int8_t *g_c2_type;

static void d2_forward(const dcomplex *v,int nt){
    /* Direct atomic accumulate into global hash (no local hashes needed) */
    #pragma omp parallel for schedule(dynamic,256) num_threads(nt)
    for(int32_t i=0;i<g_n_c2;i++){
        dcomplex vi=v[i];
        double vir=creal(vi),vii=cimag(vi);
        if(fabs(vir)<1e-30&&fabs(vii)<1e-30)continue;

        if(g_c2_type[i]==0){
            /* f-type */
            int af=g_c2_tuple[i*3],bf=g_c2_tuple[i*3+1],lf=g_c2_tuple[i*3+2];
            for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];
                c3_global_accumulate(hkey(0,mult_bar[idx].a,af,bf,mult_bar[idx].l),
                                     (scomplex)(mb_val(idx)*vi));
            }
            for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];
                c3_global_accumulate(hkey(0,mult_bar[idx].a,mult_bar[idx].b,bf,lf),
                                     (scomplex)(-mb_val(idx)*vi));
            }
            for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];
                c3_global_accumulate(hkey(0,af,mult_bar[idx].a,mult_bar[idx].b,lf),
                                     (scomplex)(mb_val(idx)*vi));
            }
            for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];
                c3_global_accumulate(hkey(0,af,bf,mult_bar[idx].b,mult_bar[idx].l),
                                     (scomplex)(-mb_val(idx)*vi));
            }
            for(int e=0;e<n_db_by_k[af];e++){int idx=db_by_k[af][e];
                c3_global_accumulate(hkey(1,delta_bar[idx].c,bf,delta_bar[idx].j,lf),
                                     (scomplex)(-db_val(idx)*vi));
            }
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];
                c3_global_accumulate(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k),
                                     (scomplex)(db_val(idx)*vi));
            }
            for(int e=0;e<n_db_by_j[bf];e++){int idx=db_by_j[bf][e];
                c3_global_accumulate(hkey(1,af,delta_bar[idx].c,lf,delta_bar[idx].k),
                                     (scomplex)(-db_val(idx)*vi));
            }
        } else {
            /* g-type */
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];
            for(int e=0;e<n_mb_by_b[jf];e++){int idx=mb_by_b[jf][e];
                c3_global_accumulate(hkey(1,mult_bar[idx].a,cf,mult_bar[idx].l,kf),
                                     (scomplex)(mb_val(idx)*vi));
            }
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];
                c3_global_accumulate(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf),
                                     (scomplex)(-mb_val(idx)*vi));
            }
            for(int e=0;e<n_mb_by_b[kf];e++){int idx=mb_by_b[kf][e];
                c3_global_accumulate(hkey(1,cf,mult_bar[idx].a,jf,mult_bar[idx].l),
                                     (scomplex)(mb_val(idx)*vi));
            }
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];
                c3_global_accumulate(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf),
                                     (scomplex)(-db_val(idx)*vi));
            }
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];
                c3_global_accumulate(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf),
                                     (scomplex)(db_val(idx)*vi));
            }
            /* -∂ᶜg T3: -db_val → hkey(2,cf,jf,j,K)  [NEW: (id⊗Δ)g term] */
            for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];
                c3_global_accumulate(hkey(2,cf,jf,delta_bar[idx].j,delta_bar[idx].k),
                                     (scomplex)(-db_val(idx)*vi));
            }
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];
                c3_global_accumulate(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k),
                                     (scomplex)(db_val(idx)*vi));
            }
        }
    }
}

/* ====================================================================== */
/*  Matrix-free d2 adjoint matvec: z = d2* @ h                            */
/*  h is stored in g_c3_global hash                                       */
/*  z: dcomplex[n_c2] (dense, output)                                     */
/* ====================================================================== */
static void d2_adjoint(dcomplex *z,int nt){
    #pragma omp parallel for schedule(dynamic,256)
    for(int32_t i=0;i<g_n_c2;i++){
        dcomplex zi=0;

        if(g_c2_type[i]==0){
            /* f-type */
            int af=g_c2_tuple[i*3],bf=g_c2_tuple[i*3+1],lf=g_c2_tuple[i*3+2];
            /* ∂ʰf T1: conj(+mb_val) → lookup hkey(0,a,af,bf,t) */
            for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];
                zi+=conj(mb_val(idx))*c3_global_lookup(hkey(0,mult_bar[idx].a,af,bf,mult_bar[idx].l));
            }
            /* ∂ʰf T2: conj(-mb_val) → lookup hkey(0,a,b,bf,lf) */
            for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];
                zi-=conj(mb_val(idx))*c3_global_lookup(hkey(0,mult_bar[idx].a,mult_bar[idx].b,bf,lf));
            }
            /* ∂ʰf T3: conj(+mb_val) → lookup hkey(0,af,b,c,lf) */
            for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];
                zi+=conj(mb_val(idx))*c3_global_lookup(hkey(0,af,mult_bar[idx].a,mult_bar[idx].b,lf));
            }
            /* ∂ʰf T4: conj(-mb_val) → lookup hkey(0,af,bf,c,t) */
            for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];
                zi-=conj(mb_val(idx))*c3_global_lookup(hkey(0,af,bf,mult_bar[idx].b,mult_bar[idx].l));
            }
            /* +∂ᶜf T1: conj(-db_val) → lookup hkey(1,a,bf,j,lf) */
            for(int e=0;e<n_db_by_k[af];e++){int idx=db_by_k[af][e];
                zi-=conj(db_val(idx))*c3_global_lookup(hkey(1,delta_bar[idx].c,bf,delta_bar[idx].j,lf));
            }
            /* +∂ᶜf T2: conj(+db_val) → lookup hkey(1,af,bf,j,k) */
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];
                zi+=conj(db_val(idx))*c3_global_lookup(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k));
            }
            /* +∂ᶜf T3: conj(-db_val) → lookup hkey(1,af,b,lf,k) */
            for(int e=0;e<n_db_by_j[bf];e++){int idx=db_by_j[bf][e];
                zi-=conj(db_val(idx))*c3_global_lookup(hkey(1,af,delta_bar[idx].c,lf,delta_bar[idx].k));
            }
        } else {
            /* g-type */
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];
            /* ∂ʰg T1: conj(+mb_val) → lookup hkey(1,a,cf,j,kf) */
            for(int e=0;e<n_mb_by_b[jf];e++){int idx=mb_by_b[jf][e];
                zi+=conj(mb_val(idx))*c3_global_lookup(hkey(1,mult_bar[idx].a,cf,mult_bar[idx].l,kf));
            }
            /* ∂ʰg T2: conj(-mb_val) → lookup hkey(1,a,b,jf,kf) */
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];
                zi-=conj(mb_val(idx))*c3_global_lookup(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf));
            }
            /* ∂ʰg T3: conj(+mb_val) → lookup hkey(1,cf,b,jf,k) */
            for(int e=0;e<n_mb_by_b[kf];e++){int idx=mb_by_b[kf][e];
                zi+=conj(mb_val(idx))*c3_global_lookup(hkey(1,cf,mult_bar[idx].a,jf,mult_bar[idx].l));
            }
            /* -∂ᶜg T1: conj(-db_val) → lookup hkey(2,c,j,jf,kf) */
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];
                zi-=conj(db_val(idx))*c3_global_lookup(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf));
            }
            /* -∂ᶜg T2: conj(+db_val) → lookup hkey(2,cf,j,K,kf) */
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];
                zi+=conj(db_val(idx))*c3_global_lookup(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf));
            }
            /* -∂ᶜg T3: conj(-db_val) → lookup hkey(2,cf,jf,j,K) */
            for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];
                zi-=conj(db_val(idx))*c3_global_lookup(hkey(2,cf,jf,delta_bar[idx].j,delta_bar[idx].k));
            }
            /* -∂ᶜg T4: conj(+db_val) → lookup hkey(2,c,jf,kf,l) */
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];
                zi+=conj(db_val(idx))*c3_global_lookup(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k));
            }
        }
        z[i]=zi;
    }
}

/* ====================================================================== */
/*  d1 matvec helpers                                                     */
/* ====================================================================== */
static spmat *g_d1;

/* w = d1* @ v  (v ∈ C², w ∈ C¹) */
static void d1_adjoint_matvec(const dcomplex *v,dcomplex *w,int32_t n_c1){
    memset(w,0,n_c1*sizeof(dcomplex));
    for(int64_t e=0;e<g_d1->nnz;e++)
        w[g_d1->cols[e]]+=conj(g_d1->vals[e])*v[g_d1->rows[e]];
}

/* z = d1 @ w  (w ∈ C¹, z ∈ C²) */
static void d1_forward_matvec(const dcomplex *w,dcomplex *z){
    memset(z,0,g_n_c2*sizeof(dcomplex));
    for(int64_t e=0;e<g_d1->nnz;e++)
        z[g_d1->rows[e]]+=g_d1->vals[e]*w[g_d1->cols[e]];
}

/* ====================================================================== */
/*  Orthonormalize columns of Q via LAPACK QR                             */
/* ====================================================================== */
static void orthonormalize(dcomplex *Q,int k,int64_t n_rows){
    int m=(int)n_rows,n=k,lda=m,info;
    dcomplex *tau=malloc(k*sizeof(dcomplex));
    int lwork=-1;dcomplex wkopt;
    zgeqrf_(&m,&n,Q,&lda,tau,&wkopt,&lwork,&info);
    lwork=(int)creal(wkopt);if(lwork<1)lwork=1;
    dcomplex *work=malloc(lwork*sizeof(dcomplex));
    zgeqrf_(&m,&n,Q,&lda,tau,work,&lwork,&info);
    zungqr_(&m,&n,&n,Q,&lda,tau,work,&lwork,&info);
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
/*  Main solve: Lanczos algorithm on A_aug* A_aug                         */
/*  Finds SMALLEST eigenvalues → detects null space                       */
/* ====================================================================== */
static int g_n_c1_global;
static int32_t n_c2_global;

/* Apply A_aug* A_aug to vector v: result = d2*(d2*v) + d1*(d1* v)
   v ∈ C², result ∈ C² */
static void apply_AAtA(const dcomplex *v, dcomplex *result, dcomplex *w_buf, dcomplex *z2_buf, int nt){
    /* h = d2 * v (stored in global hash) */
    c3_global_clear();
    d2_forward(v,nt);
    /* w = d1* * v (C¹ vector) */
    d1_adjoint_matvec(v,w_buf,g_n_c1_global);
    /* result = d2* * h (C² vector) */
    d2_adjoint(result,nt);
    /* z2 = d1 * w (C² vector) */
    d1_forward_matvec(w_buf,z2_buf);
    /* result += z2 */
    #pragma omp parallel for num_threads(nt)
    for(int32_t c=0;c<n_c2_global;c++)result[c]+=z2_buf[c];
}

static int solve(int k,int iters,int32_t n_c1,int32_t n_c2,int nt){
    g_n_c1_global=n_c1;
    n_c2_global=n_c2;
    int m=k;  /* Lanczos iterations */
    double T0=omp_get_wtime();

    printf("========================================================\n");
    printf("    Matrix-Free Lanczos Solver (pure C, AVX-512)\n");
    printf("    A_aug = [d2; d1*],  shift (0,0) only\n");
    printf("    dim H̃²_b = nullity(A_aug*A_aug)\n");
    printf("========================================================\n\n");
    printf("C¹=%d  C²=%d  Lanczos iters=%d  threads=%d\n\n",n_c1,n_c2,m,nt);
    fflush(stdout);

    /* Allocate C³ global hash */
    printf("[1] Allocating C³ global hash (%u entries, %.1f GB)...\n",
           C3_GLOBAL_SIZE,(double)C3_GLOBAL_SIZE*16/1e9);
    fflush(stdout);
    g_c3_global=malloc((size_t)C3_GLOBAL_SIZE*sizeof(c3_entry));
    if(!g_c3_global){fprintf(stderr,"malloc g_c3_global failed\n");exit(1);}
    c3_global_clear();
    printf("  done (%.1fs)\n\n",omp_get_wtime()-T0);fflush(stdout);

    /* Allocate Lanczos vectors: V[0..m], each n_c2 complex128 */
    printf("[2] Allocating Lanczos vectors (m=%d, n_c2=%d, %.1f GB)...\n",
           m,n_c2,(double)(m+1)*n_c2*16/1e9);
    fflush(stdout);
    double t=omp_get_wtime();
    dcomplex *V=malloc((size_t)(m+1)*n_c2*sizeof(dcomplex));
    dcomplex *w=malloc(n_c2*sizeof(dcomplex));
    dcomplex *w_buf=malloc(n_c1*sizeof(dcomplex));
    dcomplex *z2_buf=malloc(n_c2*sizeof(dcomplex));
    if(!V||!w||!w_buf||!z2_buf){fprintf(stderr,"malloc V/w failed\n");exit(1);}

    /* Tridiagonal matrix elements */
    double *alpha=malloc(m*sizeof(double));
    double *beta=malloc(m*sizeof(double));
    double *evals=malloc(m*sizeof(double));

    /* Lanczos vectors for reorthogonalization dot products */
    dcomplex *dots=malloc(m*sizeof(dcomplex));
    printf("  done (%.1fs)\n\n",omp_get_wtime()-t);fflush(stdout);

    /* Initialize v_0 with random vector */
    printf("[3] Starting Lanczos iteration...\n");fflush(stdout);
    t=omp_get_wtime();
    prng_t prng={{0x9E3779B97F4A7C15ULL,0xC2B2AE3D27D4EB4FULL}};
    dcomplex *v0=V;
    for(int32_t c=0;c<n_c2;c++)v0[c]=prng_cn(&prng);
    double nrm=0;
    #pragma omp parallel for reduction(+:nrm) num_threads(nt)
    for(int32_t c=0;c<n_c2;c++)nrm+=creal(conj(v0[c])*v0[c]);
    double inv_nrm=1.0/sqrt(nrm);
    #pragma omp parallel for num_threads(nt)
    for(int32_t c=0;c<n_c2;c++)v0[c]*=inv_nrm;

    beta[0]=0.0;
    dcomplex *v_prev=NULL;  /* no previous for first iteration */

    for(int j=0;j<m;j++){
        double tj=omp_get_wtime();
        dcomplex *v_j=V+j*n_c2;
        dcomplex *v_jp1=V+(j+1)*n_c2;

        /* w = A*A * v_j */
        apply_AAtA(v_j, w, w_buf, z2_buf, nt);

        /* α_j = v_j* w */
        double aj=0;
        #pragma omp parallel for reduction(+:aj) num_threads(nt)
        for(int32_t c=0;c<n_c2;c++)aj+=creal(conj(v_j[c])*w[c]);
        alpha[j]=aj;

        /* w = w - α_j v_j - β_{j-1} v_{j-1} */
        #pragma omp parallel for num_threads(nt)
        for(int32_t c=0;c<n_c2;c++){
            w[c]-=alpha[j]*v_j[c];
            if(j>0)w[c]-=beta[j-1]*v_prev[c];
        }

        /* Full reorthogonalization: w = w - sum_{i=0}^{j} (v_i* w) v_i */
        for(int i=0;i<=j;i++){
            dcomplex dot=0;
            #pragma omp parallel for reduction(+:dot) num_threads(nt)
            for(int32_t c=0;c<n_c2;c++)dot+=conj(V[i*n_c2+c])*w[c];
            #pragma omp parallel for num_threads(nt)
            for(int32_t c=0;c<n_c2;c++)w[c]-=dot*V[i*n_c2+c];
        }

        /* β_j = ||w|| */
        double bj=0;
        #pragma omp parallel for reduction(+:bj) num_threads(nt)
        for(int32_t c=0;c<n_c2;c++)bj+=creal(conj(w[c])*w[c]);
        bj=sqrt(bj);
        beta[j]=bj;

        /* v_{j+1} = w / β_j */
        if(bj>1e-30){
            double inv=1.0/bj;
            #pragma omp parallel for num_threads(nt)
            for(int32_t c=0;c<n_c2;c++)v_jp1[c]=w[c]*inv;
        } else {
            /* β_j ≈ 0: invariant subspace found, generate new random vector */
            printf("  ** β_%d ≈ 0, invariant subspace found! **\n",j);
            for(int32_t c=0;c<n_c2;c++)v_jp1[c]=prng_cn(&prng);
            /* Reorthogonalize against all previous */
            for(int i=0;i<=j;i++){
                dcomplex dot=0;
                #pragma omp parallel for reduction(+:dot) num_threads(nt)
                for(int32_t c=0;c<n_c2;c++)dot+=conj(V[i*n_c2+c])*v_jp1[c];
                #pragma omp parallel for num_threads(nt)
                for(int32_t c=0;c<n_c2;c++)v_jp1[c]-=dot*V[i*n_c2+c];
            }
            double nn=0;
            #pragma omp parallel for reduction(+:nn) num_threads(nt)
            for(int32_t c=0;c<n_c2;c++)nn+=creal(conj(v_jp1[c])*v_jp1[c]);
            double inv=sqrt(nn);
            #pragma omp parallel for num_threads(nt)
            for(int32_t c=0;c<n_c2;c++)v_jp1[c]*=inv;
        }

        v_prev=v_j;

        /* Compute eigenvalues of T_j (tridiagonal) every few iterations */
        if((j+1)%5==0||j==m-1||bj<1e-10){
            int nj=j+1;
            /* Build tridiagonal matrix and compute eigenvalues via dstev */
            double *Tdiag=malloc(nj*sizeof(double));
            double *Toff=malloc((nj-1)*sizeof(double));
            double *Tevals=malloc(nj*sizeof(double));
            double *Twork=malloc((2*nj-2)*sizeof(double));
            memcpy(Tdiag,alpha,nj*sizeof(double));
            memcpy(Toff,beta,(nj-1)*sizeof(double));

            /* dstev_: LAPACK symmetric tridiagonal eigenvalue solver
               Signature: dstev(JOBZ, N, D, E, Z, LDZ, WORK, INFO) */
            extern void dstev_(const char*,const int*,double*,double*,double*,const int*,double*,int*);
            int info;
            int ldz=1;
            dstev_("N",&nj,Tdiag,Toff,Tevals,&ldz,Twork,&info);
            if(info!=0){
                printf("  dstev failed: info=%d\n",info);
            } else {
                /* dstev overwrites D (Tdiag) with eigenvalues in ascending order */
                printf("  iter %3d/%d (%5.0fs): smallest Ritz = [",j+1,m,omp_get_wtime()-tj);
                for(int i=0;i<nj&&i<8;i++)printf("%.3e ",Tdiag[i]);
                printf("]\n");fflush(stdout);

                /* Check for convergence: if smallest eigenvalue is stable, stop early */
                if(j+1>=20){
                    double min_eig=Tdiag[0];
                    double lam_max_est=Tdiag[nj-1];
                    printf("    λ_min=%.6e  λ_max=%.6e  ratio=%.6e\n",
                           min_eig,lam_max_est,min_eig/lam_max_est);
                    fflush(stdout);
                }
            }
            free(Tdiag);free(Toff);free(Tevals);free(Twork);
        } else {
            printf("  iter %3d/%d (%5.0fs)  α=%.4e  β=%.4e\n",
                   j+1,m,omp_get_wtime()-tj,alpha[j],beta[j]);fflush(stdout);
        }
    }
    printf("  total: %.0fs (%.1f min)\n\n",omp_get_wtime()-t,(omp_get_wtime()-t)/60.0);fflush(stdout);

    /* Final eigenvalue computation */
    printf("[4] FINAL RESULT\n\n");
    int nj=m;
    double *Tdiag=malloc(nj*sizeof(double));
    double *Toff=malloc((nj-1)*sizeof(double));
    double *Tevals=malloc(nj*sizeof(double));
    double *Twork=malloc((2*nj-2)*sizeof(double));
    memcpy(Tdiag,alpha,nj*sizeof(double));
    memcpy(Toff,beta,(nj-1)*sizeof(double));
    extern void dstev_(const char*,const int*,double*,double*,double*,const int*,double*,int*);
    int info;
    int ldz=1;
    dstev_("N",&nj,Tdiag,Toff,Tevals,&ldz,Twork,&info);

    printf("  Ritz values (ascending):\n");
    for(int i=0;i<nj&&i<20;i++)
        printf("    λ[%2d] = %+.10e\n",i,Tdiag[i]);
    printf("\n");

    /* Nullity detection: count near-zero eigenvalues */
    double lam_max_est=Tdiag[nj-1];
    double tol=lam_max_est*1e-8;
    int nullity=0;
    for(int i=0;i<nj;i++){
        if(Tdiag[i]<tol)nullity++;
        else break;
    }
    /* Also check for largest gap */
    int best_gap=-1;double best_log=-1e9;
    for(int i=1;i<nj;i++){
        if(Tdiag[i-1]>0&&Tdiag[i]>0){
            double lg=log10(Tdiag[i]/Tdiag[i-1]);
            if(lg>best_log){best_log=lg;best_gap=i;}
        }
    }
    printf("  λ_max ≈ %.6e\n",lam_max_est);
    printf("  Largest gap: λ[%d]/λ[%d] (log10=%.2f)\n",best_gap,best_gap-1,best_log);
    printf("  Near-zero count (tol=%.2e): %d\n",tol,nullity);
    printf("\n  ============================================\n");
    printf("  ||  NULLITY (per shift) = %-3d              ||\n",nullity);
    printf("  ||  NULLITY (total, ×3) = %-3d              ||\n",nullity*3);
    printf("  ||  dim H̃²_b = %-3d                        ||\n",nullity*3);
    printf("  ||  dim HH²(sl₃) = 12 - %-3d = %-3d          ||\n",nullity*3,12-nullity*3);
    printf("  ============================================\n\n");fflush(stdout);

    free(Tdiag);free(Toff);free(Tevals);free(Twork);
    free(V);free(w);free(w_buf);free(z2_buf);
    free(alpha);free(beta);free(evals);free(dots);
    free(g_c3_global);

    printf("Total time: %.0fs (%.1f min)\n",omp_get_wtime()-T0,(omp_get_wtime()-T0)/60.0);fflush(stdout);
    return nullity;
}

/* ====================================================================== */
/*  Main                                                                  */
/* ====================================================================== */
int main(int argc,char*argv[]){
    const char*dir=(argc>1)?argv[1]:".";
    int k=(argc>2)?atoi(argv[2]):20;
    int iters=(argc>3)?atoi(argv[3]):20;
    int nt=omp_get_max_threads();
    if(nt>15)nt=15;
    omp_set_num_threads(nt);
    double T0=omp_get_wtime();

    printf("=== sl_3 HH² solver (pure C, AVX-512, OpenMP) ===\nThreads=%d\n\n",nt);fflush(stdout);

    /* Phase 1: Load algebra */
    printf("[A] Loading algebra...\n");fflush(stdout);
    load_algebra(dir);build_weight_classes();
    printf("  mult_bar:%d delta_bar:%d\n",NNZ_MB,NNZ_DB);
    for(int k2=0;k2<ELL*ELL;k2++)if(w2i_n[k2])printf("  wt(%d,%d):%d\n",k2/ELL,k2%ELL,w2i_n[k2]);
    fflush(stdout);

    /* Phase 2: Build C¹, C², d1 */
    printf("\n[B] Building C1, C2, d1...\n");fflush(stdout);
    double t=omp_get_wtime();
    int s0=0,s1=0;

    /* C1 */
    int32_t n_c1=0;int32_t c1j[20000],c1k[20000];
    for(int wi=0;wi<w2i_nkeys;wi++){int wj=w2i_order[wi];
    for(int wj2=0;wj2<w2i_nkeys;wj2++){int wk=w2i_order[wj2];
        if((wj/ELL-wk/ELL)%ELL==s0&&(wj%ELL-wk%ELL)%ELL==s1)
            for(int ji=0;ji<w2i_n[wj];ji++)for(int ki=0;ki<w2i_n[wk];ki++)
                {c1j[n_c1]=w2i_elems[wj][ji];c1k[n_c1]=w2i_elems[wk][ki];n_c1++;}
    }}

    /* C2 */
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

    /* Build d1 */
    printf("[C] Building d1...\n");fflush(stdout);
    t=omp_get_wtime();
    spmat*d1=spmat_new(n_c2,n_c1,10000000);
    for(int ci=0;ci<n_c1;ci++){
        int hj=c1j[ci],hk=c1k[ci];
        for(int i=0;i<n_mb_by_b[hk];i++){int idx=mb_by_b[hk][i];
            int32_t r=hash_insert(fkey(mult_bar[idx].a,hk,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
        for(int i=0;i<n_mb_by_l[hk];i++){int idx=mb_by_l[hk][i];
            int32_t r=hash_insert(fkey(mult_bar[idx].a,mult_bar[idx].b,hj));
            if(r>=0)spmat_add(d1,r,ci,-mb_val(idx));}
        for(int i=0;i<n_mb_by_a[hk];i++){int idx=mb_by_a[hk][i];
            int32_t r=hash_insert(fkey(hk,mult_bar[idx].b,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
        for(int i=0;i<n_db_by_k[hk];i++){int idx=db_by_k[hk][i];
            int32_t r=hash_insert(gkey(delta_bar[idx].c,delta_bar[idx].j,hj));
            if(r>=0)spmat_add(d1,r,ci,-db_val(idx));}
        for(int i=0;i<n_db_by_c[hk];i++){int idx=db_by_c[hk][i];
            int32_t r=hash_insert(gkey(hk,delta_bar[idx].j,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,db_val(idx));}
        for(int i=0;i<n_db_by_j[hk];i++){int idx=db_by_j[hk][i];
            int32_t r=hash_insert(gkey(delta_bar[idx].c,hj,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,-db_val(idx));}
    }
    printf("  d1: %lld nnz (%.1fs)\n\n",(long long)d1->nnz,omp_get_wtime()-t);fflush(stdout);

    /* Set global pointers for solver */
    g_n_c2=n_c2;
    g_c2_tuple=c2_tuple;
    g_c2_type=c2_type;
    g_d1=d1;

    /* Phase 3: Matrix-free augmented solver */
    printf("[D] Starting matrix-free augmented solver...\n\n");fflush(stdout);
    int nullity=solve(k,iters,n_c1,n_c2,nt);

    printf("\n=== FINAL RESULT ===\n");
    printf("  nullity per shift = %d\n",nullity);
    printf("  dim H̃²_b(B⁺) = %d (×3 shifts)\n",nullity*3);
    printf("  dim HH²(u_q(sl_3)) = 12 - %d = %d\n",nullity*3,12-nullity*3);
    printf("\nTotal: %.1fs (%.1f min)\n",omp_get_wtime()-T0,(omp_get_wtime()-T0)/60.0);fflush(stdout);

    spmat_free(d1);
    free(c2hash);
    return 0;
}
