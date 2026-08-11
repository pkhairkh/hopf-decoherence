/*
 * sl3_projected_lanczos.c — Projected Lanczos for nullity(M) where M = d2*d2 + d1*d1*.
 *
 * KEY INSIGHT: M is a Hodge Laplacian. ker(d1*) is invariant under M.
 * By starting Lanczos in ker(d1*), we exclude the 19520 large eigenvalues
 * from d1*d1* and explore only the spectrum of d2*d2.
 *
 * Projection: v0 = w - d1 @ (d1*d1)^{-1} @ (d1* w)
 *   - d1*d1 is 19522×19522 (already eigendecomposed with zheevd)
 *   - Cost: one d1* matvec (~1s) + one 19522×19522 solve (~0.1s) + one d1 matvec (~1s)
 *   - Total projection cost: ~2s
 *
 * Then standard Lanczos for 15-20 iterations (~250s each = 1-1.5 hours)
 *
 * Pure C. AVX-512. OpenMP. Householder QR reorthogonalization.
 *
 * Build:
 *   gcc -O3 -march=znver5 -mtune=znver5 -fopenmp -fno-math-errno -mcx16 \
 *       -o sl3_projected_lanczos sl3_projected_lanczos.c \
 *       -L/usr/lib64 -lopenblas -lgfortran -lm
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
extern void zheev_(const char*,const char*,const int*,dcomplex*,const int*,double*,dcomplex*,const int*,double*,int*);
extern void zgemm_(const char*,const char*,const int*,const int*,const int*,const dcomplex*,const dcomplex*,const int*,const dcomplex*,const int*,const dcomplex*,dcomplex*,const int*);
extern void zpotrf_(const char*,const int*,dcomplex*,const int*,int*);
extern void zpotrs_(const char*,const int*,const int*,dcomplex*,const int*,dcomplex*,const int*,int*);

/* ====================================================================== */
/*  Algebra data (same as sl3_tracemin.c)                                 */
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

/* C³ hash */
#define C3_BITS 31
#define C3_SIZE (1u<<C3_BITS)
#define C3_MASK (C3_SIZE-1)
typedef struct{uint64_t key;scomplex val;}c3_entry;
static c3_entry *g_c3;
static int32_t g_n_c2;
static int32_t *g_c2_tuple;
static int8_t *g_c2_type;
static spmat *g_d1;
static int g_n_c1;

static inline uint32_t c3_pos(uint64_t key){
    key^=key>>33;key*=0xff51afd7ed558ccdULL;key^=key>>33;
    return (uint32_t)(key&C3_MASK);
}
static inline uint64_t hkey(int comp,int a,int b,int c,int d){
    return ((uint64_t)comp<<60)|((uint64_t)a<<48)|((uint64_t)b<<32)|((uint64_t)c<<16)|((uint64_t)d);
}
static void c3_clear(void){
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<(int64_t)C3_SIZE;i++)g_c3[i].key=0xFFFFFFFFFFFFFFFFULL;
}
static inline void c3_accum(uint64_t key,scomplex val){
    uint32_t pos=c3_pos(key);
    while(1){
        uint64_t cur=g_c3[pos].key;
        if(cur==0xFFFFFFFFFFFFFFFFULL){
            if(__sync_bool_compare_and_swap(&g_c3[pos].key,cur,key)){g_c3[pos].val=val;return;}
            continue;
        }
        if(cur==key){
            uint64_t*vp=(uint64_t*)&g_c3[pos].val;uint64_t old_v,new_v;
            do{old_v=*vp;scomplex oc=*(scomplex*)&old_v;scomplex nc=oc+val;new_v=*(uint64_t*)&nc;}while(!__sync_bool_compare_and_swap(vp,old_v,new_v));
            return;
        }
        pos=(pos+1)&C3_MASK;
    }
}
static inline dcomplex c3_lookup(uint64_t key){
    uint32_t pos=c3_pos(key);
    while(g_c3[pos].key!=0xFFFFFFFFFFFFFFFFULL){
        if(g_c3[pos].key==key)return (dcomplex)g_c3[pos].val;
        pos=(pos+1)&C3_MASK;
    }
    return 0.0;
}

/* d2_forward and d2_adjoint (same corrected MW formula as sl3_tracemin.c) */
static void d2_forward(const dcomplex *v,int nt){
    #pragma omp parallel for schedule(dynamic,64) num_threads(nt)
    for(int32_t i=0;i<g_n_c2;i++){
        dcomplex vi=v[i];
        if(cabs(vi)<1e-30)continue;
        if(g_c2_type[i]==0){
            int af=g_c2_tuple[i*3],bf=g_c2_tuple[i*3+1],lf=g_c2_tuple[i*3+2];
            for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];c3_accum(hkey(0,mult_bar[idx].a,af,bf,mult_bar[idx].l),(scomplex)(mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];c3_accum(hkey(0,mult_bar[idx].a,mult_bar[idx].b,bf,lf),(scomplex)(-mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];c3_accum(hkey(0,af,mult_bar[idx].a,mult_bar[idx].b,lf),(scomplex)(mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];c3_accum(hkey(0,af,bf,mult_bar[idx].b,mult_bar[idx].l),(scomplex)(-mb_val(idx)*vi));}
            for(int e1=0;e1<n_db_by_k[af];e1++){int d1idx=db_by_k[af][e1];int a=delta_bar[d1idx].c,a1=delta_bar[d1idx].j;dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_k[bf];e2++){int d2idx=db_by_k[bf][e2];int b=delta_bar[d2idx].c,b1=delta_bar[d2idx].j;dcomplex d2v=db_val(d2idx);
                    int ab=a1*DIM+b1;for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];c3_accum(hkey(1,a,b,mult_bar[midx].l,lf),(scomplex)(d1v*d2v*mb_val(midx)*vi));}}}
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];c3_accum(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k),(scomplex)(-db_val(idx)*vi));}
            for(int e1=0;e1<n_db_by_j[af];e1++){int d1idx=db_by_j[af][e1];int a=delta_bar[d1idx].c,a2=delta_bar[d1idx].k;dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_j[bf];e2++){int d2idx=db_by_j[bf][e2];int b=delta_bar[d2idx].c,b2=delta_bar[d2idx].k;dcomplex d2v=db_val(d2idx);
                    int ab=a2*DIM+b2;for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];c3_accum(hkey(1,a,b,lf,mult_bar[midx].l),(scomplex)(d1v*d2v*mb_val(midx)*vi));}}}
        } else {
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];
            for(int e1=0;e1<n_mb_by_b[jf];e1++){int m1idx=mb_by_b[jf][e1];int t1=mult_bar[m1idx].l,a1=mult_bar[m1idx].a;dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_b[kf];e2++){int m2idx=mb_by_b[kf][e2];int t2=mult_bar[m2idx].l,a2=mult_bar[m2idx].a;dcomplex m2v=mb_val(m2idx);
                    int jk=a1*DIM+a2;for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];c3_accum(hkey(1,delta_bar[didx].c,cf,t1,t2),(scomplex)(db_val(didx)*m1v*m2v*vi));}}}
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];c3_accum(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf),(scomplex)(-mb_val(idx)*vi));}
            for(int e1=0;e1<n_mb_by_a[jf];e1++){int m1idx=mb_by_a[jf][e1];int t1=mult_bar[m1idx].l,b1=mult_bar[m1idx].b;dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_a[kf];e2++){int m2idx=mb_by_a[kf][e2];int t2=mult_bar[m2idx].l,b2=mult_bar[m2idx].b;dcomplex m2v=mb_val(m2idx);
                    int jk=b1*DIM+b2;for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];c3_accum(hkey(1,cf,delta_bar[didx].c,t1,t2),(scomplex)(m1v*m2v*db_val(didx)*vi));}}}
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];c3_accum(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf),(scomplex)(-db_val(idx)*vi));}
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];c3_accum(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf),(scomplex)(db_val(idx)*vi));}
            for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];c3_accum(hkey(2,cf,jf,delta_bar[idx].j,delta_bar[idx].k),(scomplex)(-db_val(idx)*vi));}
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];c3_accum(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k),(scomplex)(db_val(idx)*vi));}
        }
    }
}

static void d2_adjoint(dcomplex *z,int nt){
    #pragma omp parallel for schedule(dynamic,64) num_threads(nt)
    for(int32_t i=0;i<g_n_c2;i++){
        dcomplex zi=0;
        if(g_c2_type[i]==0){
            int af=g_c2_tuple[i*3],bf=g_c2_tuple[i*3+1],lf=g_c2_tuple[i*3+2];
            for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];zi+=conj(mb_val(idx))*c3_lookup(hkey(0,mult_bar[idx].a,af,bf,mult_bar[idx].l));}
            for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];zi-=conj(mb_val(idx))*c3_lookup(hkey(0,mult_bar[idx].a,mult_bar[idx].b,bf,lf));}
            for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];zi+=conj(mb_val(idx))*c3_lookup(hkey(0,af,mult_bar[idx].a,mult_bar[idx].b,lf));}
            for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];zi-=conj(mb_val(idx))*c3_lookup(hkey(0,af,bf,mult_bar[idx].b,mult_bar[idx].l));}
            for(int e1=0;e1<n_db_by_k[af];e1++){int d1idx=db_by_k[af][e1];int a=delta_bar[d1idx].c,a1=delta_bar[d1idx].j;dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_k[bf];e2++){int d2idx=db_by_k[bf][e2];int b=delta_bar[d2idx].c,b1=delta_bar[d2idx].j;dcomplex d2v=db_val(d2idx);
                    int ab=a1*DIM+b1;for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];zi+=conj(d1v*d2v*mb_val(midx))*c3_lookup(hkey(1,a,b,mult_bar[midx].l,lf));}}}
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];zi-=conj(db_val(idx))*c3_lookup(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k));}
            for(int e1=0;e1<n_db_by_j[af];e1++){int d1idx=db_by_j[af][e1];int a=delta_bar[d1idx].c,a2=delta_bar[d1idx].k;dcomplex d1v=db_val(d1idx);
                for(int e2=0;e2<n_db_by_j[bf];e2++){int d2idx=db_by_j[bf][e2];int b=delta_bar[d2idx].c,b2=delta_bar[d2idx].k;dcomplex d2v=db_val(d2idx);
                    int ab=a2*DIM+b2;for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];zi+=conj(d1v*d2v*mb_val(midx))*c3_lookup(hkey(1,a,b,lf,mult_bar[midx].l));}}}
        } else {
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];
            for(int e1=0;e1<n_mb_by_b[jf];e1++){int m1idx=mb_by_b[jf][e1];int t1=mult_bar[m1idx].l,a1=mult_bar[m1idx].a;dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_b[kf];e2++){int m2idx=mb_by_b[kf][e2];int t2=mult_bar[m2idx].l,a2=mult_bar[m2idx].a;dcomplex m2v=mb_val(m2idx);
                    int jk=a1*DIM+a2;for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];zi+=conj(m1v*m2v*db_val(didx))*c3_lookup(hkey(1,delta_bar[didx].c,cf,t1,t2));}}}
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];zi-=conj(mb_val(idx))*c3_lookup(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf));}
            for(int e1=0;e1<n_mb_by_a[jf];e1++){int m1idx=mb_by_a[jf][e1];int t1=mult_bar[m1idx].l,b1=mult_bar[m1idx].b;dcomplex m1v=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_a[kf];e2++){int m2idx=mb_by_a[kf][e2];int t2=mult_bar[m2idx].l,b2=mult_bar[m2idx].b;dcomplex m2v=mb_val(m2idx);
                    int jk=b1*DIM+b2;for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];zi+=conj(m1v*m2v*db_val(didx))*c3_lookup(hkey(1,cf,delta_bar[didx].c,t1,t2));}}}
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];zi-=conj(db_val(idx))*c3_lookup(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf));}
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];zi+=conj(db_val(idx))*c3_lookup(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf));}
            for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];zi-=conj(db_val(idx))*c3_lookup(hkey(2,cf,jf,delta_bar[idx].j,delta_bar[idx].k));}
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];zi+=conj(db_val(idx))*c3_lookup(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k));}
        }
        z[i]=zi;
    }
}

/* d1 forward and adjoint */
static void d1_forward(const dcomplex *w,dcomplex *z){
    memset(z,0,g_n_c2*sizeof(dcomplex));
    for(int64_t e=0;e<g_d1->nnz;e++)z[g_d1->rows[e]]+=g_d1->vals[e]*w[g_d1->cols[e]];
}
static void d1_adjoint(const dcomplex *v,dcomplex *w){
    memset(w,0,g_n_c1*sizeof(dcomplex));
    for(int64_t e=0;e<g_d1->nnz;e++)w[g_d1->cols[e]]+=conj(g_d1->vals[e])*v[g_d1->rows[e]];
}

/* Apply M = d2*d2 + d1*d1* to vector v */
static void apply_M(const dcomplex *v, dcomplex *result, dcomplex *w_buf, dcomplex *z2_buf, int nt){
    c3_clear();
    d2_forward(v,nt);
    d1_adjoint(v,w_buf);
    d2_adjoint(result,nt);
    d1_forward(w_buf,z2_buf);
    #pragma omp parallel for num_threads(nt)
    for(int32_t c=0;c<g_n_c2;c++)result[c]+=z2_buf[c];
}

/* Householder QR */
static void householder_qr(dcomplex *Q, int k, int64_t n){
    int m=(int)n,ncol=k,lda=m,info;
    dcomplex *tau=malloc(k*sizeof(dcomplex));
    int lwork=-1;dcomplex wkopt;
    zgeqrf_(&m,&ncol,Q,&lda,tau,&wkopt,&lwork,&info);
    lwork=(int)creal(wkopt);if(lwork<1)lwork=1;
    dcomplex *work=malloc(lwork*sizeof(dcomplex));
    zgeqrf_(&m,&ncol,Q,&lda,tau,work,&lwork,&info);
    zungqr_(&m,&ncol,&ncol,Q,&lda,tau,work,&lwork,&info);
    free(work);free(tau);
}

/* PRNG */
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
/*  Projected Lanczos                                                     */
/*  1. Project random v0 onto ker(d1*) using d1@(d1*d1)^{-1}@(d1* v)    */
/*  2. Run standard Lanczos with Householder QR reorthogonalization      */
/*  3. M automatically stays in ker(d1*) — explores only d2*d2 spectrum */
/* ====================================================================== */
static int solve_projected_lanczos(int k_iters, int nt){
    double T0=omp_get_wtime();
    int32_t n_c2=g_n_c2;
    int32_t n_c1=g_n_c1;

    printf("========================================================\n");
    printf("    Projected Lanczos (Householder QR, ker(d1*))       \n");
    printf("    M = d2*d2 + d1*d1*  (Hodge Laplacian)             \n");
    printf("    Project v0 onto ker(d1*) to exclude d1*d1* spectrum\n");
    printf("========================================================\n\n");
    printf("C¹=%d  C²=%d  Lanczos iters=%d  threads=%d\n\n",n_c1,n_c2,k_iters,nt);
    fflush(stdout);

    /* Allocate C³ hash */
    printf("[1] Allocating C³ hash (%u entries, %.1f GB)...\n",
           C3_SIZE,(double)C3_SIZE*sizeof(c3_entry)/1e9);fflush(stdout);
    g_c3=malloc((size_t)C3_SIZE*sizeof(c3_entry));
    if(!g_c3){fprintf(stderr,"malloc g_c3 failed\n");exit(1);}
    madvise(g_c3,(size_t)C3_SIZE*sizeof(c3_entry),MADV_HUGEPAGE);
    c3_clear();
    printf("  done (%.1fs)\n\n",omp_get_wtime()-T0);fflush(stdout);

    /* Step 1: Build Gram matrix G = d1*d1* (n_c1 × n_c1 = 19522 × 19522) */
    printf("[2] Building Gram matrix G = d1*d1* (%d × %d, %.2f GB)...\n",
           n_c1,n_c1,(double)n_c1*n_c1*16/1e9);fflush(stdout);
    double t=omp_get_wtime();
    dcomplex *Gmat=calloc((size_t)n_c1*n_c1,sizeof(dcomplex));
    if(!Gmat){fprintf(stderr,"calloc Gmat failed\n");exit(1);}

    /* Sort d1 by row for Gram computation */
    int32_t *row_cnt=calloc(n_c2,sizeof(int32_t));
    for(int64_t e=0;e<g_d1->nnz;e++)row_cnt[g_d1->rows[e]]++;
    int32_t *row_off=calloc(n_c2,sizeof(int32_t));
    int32_t acc=0;
    for(int r=0;r<n_c2;r++){row_off[r]=acc;acc+=row_cnt[r];}
    int32_t *sorted_cols=malloc(g_d1->nnz*sizeof(int32_t));
    dcomplex *sorted_vals=malloc(g_d1->nnz*sizeof(dcomplex));
    int32_t *pos=calloc(n_c2,sizeof(int32_t));
    for(int64_t e=0;e<g_d1->nnz;e++){
        int32_t r=g_d1->rows[e];
        int32_t p=row_off[r]+pos[r]++;
        sorted_cols[p]=g_d1->cols[e];
        sorted_vals[p]=g_d1->vals[e];
    }

    /* G[i,j] = sum_e conj(val_e[i]) * val_e[j] for entries in same row */
    for(int32_t r=0;r<n_c2;r++){
        int32_t start=row_off[r],cnt=row_cnt[r];
        if(cnt==0)continue;
        for(int32_t i=0;i<cnt;i++){
            int32_t ci=sorted_cols[start+i];
            dcomplex vi=sorted_vals[start+i];
            for(int32_t j=0;j<cnt;j++){
                int32_t cj=sorted_cols[start+j];
                dcomplex vj=sorted_vals[start+j];
                Gmat[ci+(int64_t)cj*n_c1]+=conj(vi)*vj;
            }
        }
    }
    free(row_cnt);free(row_off);free(sorted_cols);free(sorted_vals);free(pos);
    printf("  Gram built (%.1fs)\n",omp_get_wtime()-t);fflush(stdout);

    /* Cholesky factorize G = L L* */
    printf("  Cholesky factorizing G...\n");fflush(stdout);
    t=omp_get_wtime();
    int info;
    zpotrf_("L",&n_c1,Gmat,&n_c1,&info);
    if(info!=0){fprintf(stderr,"zpotrf failed: info=%d\n",info);exit(1);}
    printf("  done (%.1fs)\n\n",omp_get_wtime()-t);fflush(stdout);

    /* Step 2: Generate random vector and project onto ker(d1*) */
    printf("[3] Projecting random vector onto ker(d1*)...\n");fflush(stdout);
    t=omp_get_wtime();

    dcomplex *v0=malloc(n_c2*sizeof(dcomplex));
    dcomplex *w_buf=malloc(n_c1*sizeof(dcomplex));
    dcomplex *z2_buf=malloc(n_c2*sizeof(dcomplex));
    dcomplex *mv_buf=malloc(n_c2*sizeof(dcomplex));

    prng_t prng={{0x9E3779B97F4A7C15ULL,0xC2B2AE3D27D4EB4FULL}};
    for(int32_t c=0;c<n_c2;c++)v0[c]=prng_cn(&prng);

    /* w = d1* v0  (n_c1 vector) */
    d1_adjoint(v0,w_buf);

    /* Solve G x = w  →  x = (d1*d1*)^{-1} (d1* v0)  via Cholesky */
    int nrhs=1;
    zpotrs_("L",&n_c1,&nrhs,Gmat,&n_c1,w_buf,&n_c1,&info);
    if(info!=0){fprintf(stderr,"zpotrs failed: info=%d\n",info);exit(1);}

    /* v0 = v0 - d1 @ x  (now v0 ∈ ker(d1*)) */
    d1_forward(w_buf,z2_buf);
    #pragma omp parallel for num_threads(nt)
    for(int32_t c=0;c<n_c2;c++)v0[c]-=z2_buf[c];

    /* Normalize */
    double nrm=0;
    #pragma omp parallel for reduction(+:nrm) num_threads(nt)
    for(int32_t c=0;c<n_c2;c++)nrm+=creal(conj(v0[c])*v0[c]);
    double inv=1.0/sqrt(nrm);
    #pragma omp parallel for num_threads(nt)
    for(int32_t c=0;c<n_c2;c++)v0[c]*=inv;

    printf("  projected (||v0||=%.6e, projection took %.1fs)\n\n",sqrt(nrm),omp_get_wtime()-t);fflush(stdout);

    /* Verify projection: ||d1* v0|| should be ~0 */
    d1_adjoint(v0,w_buf);
    double d1star_norm=0;
    for(int i=0;i<n_c1;i++)d1star_norm+=creal(conj(w_buf[i])*w_buf[i]);
    printf("  ||d1* v0|| = %.6e (should be ~0)\n\n",sqrt(d1star_norm));fflush(stdout);

    /* Step 3: Lanczos iteration */
    printf("[4] Lanczos iteration (k=%d, projected onto ker(d1*)):\n\n",k_iters);fflush(stdout);

    /* Allocate Lanczos vectors V[0..k] (each n_c2) */
    dcomplex *V=malloc((size_t)(k_iters+1)*n_c2*sizeof(dcomplex));
    dcomplex *w=malloc(n_c2*sizeof(dcomplex));
    double *alpha=malloc(k_iters*sizeof(double));
    double *beta=malloc(k_iters*sizeof(double));

    /* v0 already in V[0] */
    memcpy(V,v0,n_c2*sizeof(dcomplex));
    beta[0]=0;

    /* Full reorthogonalization: store all V vectors and reorthogonalize each step */
    dcomplex *dots=malloc(k_iters*sizeof(dcomplex));

    for(int j=0;j<k_iters;j++){
        double tj=omp_get_wtime();
        dcomplex *v_j=V+j*n_c2;
        dcomplex *v_jp1=V+(j+1)*n_c2;

        /* w = M @ v_j */
        apply_M(v_j,w,w_buf,z2_buf,nt);

        /* α_j = v_j* w */
        double aj=0;
        #pragma omp parallel for reduction(+:aj) num_threads(nt)
        for(int32_t c=0;c<n_c2;c++)aj+=creal(conj(v_j[c])*w[c]);
        alpha[j]=aj;

        /* w = w - α_j v_j - β_{j-1} v_{j-1} */
        #pragma omp parallel for num_threads(nt)
        for(int32_t c=0;c<n_c2;c++){
            w[c]-=alpha[j]*v_j[c];
            if(j>0)w[c]-=beta[j-1]*V[(j-1)*n_c2+c];
        }

        /* Full reorthogonalization */
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

        if(bj>1e-30){
            double inv_bj=1.0/bj;
            #pragma omp parallel for num_threads(nt)
            for(int32_t c=0;c<n_c2;c++)v_jp1[c]=w[c]*inv_bj;
        } else {
            printf("  ** β_%d ≈ 0, invariant subspace found! **\n",j);
            for(int32_t c=0;c<n_c2;c++)v_jp1[c]=prng_cn(&prng);
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
            double inv_nn=1.0/sqrt(nn);
            #pragma omp parallel for num_threads(nt)
            for(int32_t c=0;c<n_c2;c++)v_jp1[c]*=inv_nn;
        }

        /* Compute Ritz values every 5 iterations or last */
        if((j+1)%5==0||j==k_iters-1||bj<1e-10){
            int nj=j+1;
            double *Tdiag=malloc(nj*sizeof(double));
            double *Toff=malloc((nj-1)*sizeof(double));
            double *Tevals=malloc(nj*sizeof(double));
            double *Twork=malloc((2*nj-2)*sizeof(double));
            memcpy(Tdiag,alpha,nj*sizeof(double));
            memcpy(Toff,beta,(nj-1)*sizeof(double));
            extern void dstev_(const char*,const int*,double*,double*,double*,const int*,double*,int*);
            int ldz=1,info2;
            dstev_("N",&nj,Tdiag,Toff,Tevals,&ldz,Twork,&info2);
            if(info2==0){
                printf("  iter %3d/%d (%5.0fs): Ritz=[",j+1,k_iters,omp_get_wtime()-tj);
                for(int i=0;i<nj&&i<8;i++)printf("%.3e ",Tdiag[i]);
                printf("]\n");fflush(stdout);
                if(j+1>=10){
                    printf("    λ_min=%.6e  λ_max=%.6e  ratio=%.6e\n",
                           Tdiag[0],Tdiag[nj-1],Tdiag[0]/Tdiag[nj-1]);
                    fflush(stdout);
                }
            }
            free(Tdiag);free(Toff);free(Tevals);free(Twork);
        } else {
            printf("  iter %3d/%d (%5.0fs)  α=%.4e  β=%.4e\n",
                   j+1,k_iters,omp_get_wtime()-tj,alpha[j],beta[j]);fflush(stdout);
        }
    }

    /* Final result */
    printf("\n[5] FINAL RESULT\n\n");
    int nj=k_iters;
    double *Tdiag=malloc(nj*sizeof(double));
    double *Toff=malloc((nj-1)*sizeof(double));
    double *Twork=malloc((2*nj-2)*sizeof(double));
    memcpy(Tdiag,alpha,nj*sizeof(double));
    memcpy(Toff,beta,(nj-1)*sizeof(double));
    extern void dstev_(const char*,const int*,double*,double*,double*,const int*,double*,int*);
    int ldz=1,info2;
    dstev_("N",&nj,Tdiag,Toff,Tdiag,&ldz,Twork,&info2);

    printf("  Ritz values (ascending):\n");
    for(int i=0;i<nj&&i<20;i++)printf("    λ[%2d] = %+.10e\n",i,Tdiag[i]);
    printf("\n");

    double lam_max=Tdiag[nj-1];
    double tol=lam_max*1e-6;
    int nullity=0;
    for(int i=0;i<nj;i++){
        if(Tdiag[i]<tol)nullity++;
        else break;
    }
    int best_gap=-1;double best_log=-1e9;
    for(int i=1;i<nj;i++){
        if(Tdiag[i-1]>0&&Tdiag[i]>0){
            double lg=log10(Tdiag[i]/Tdiag[i-1]);
            if(lg>best_log){best_log=lg;best_gap=i;}
        }
    }
    printf("  λ_max ≈ %.6e\n",lam_max);
    printf("  Largest gap: λ[%d]/λ[%d] (log10=%.2f)\n",best_gap,best_gap-1,best_log);
    printf("  Near-zero count (tol=%.2e): %d\n",tol,nullity);
    printf("\n  ============================================\n");
    printf("  ||  NULLITY (per shift) = %-3d              ||\n",nullity);
    printf("  ||  dim H̃²_b(B⁺) = %-3d                    ||\n",nullity);
    printf("  ||  dim HH²(sl₃) = 9 - %-3d = %-3d          ||\n",nullity,9-nullity);
    printf("  ============================================\n\n");fflush(stdout);

    free(Tdiag);free(Toff);free(Twork);
    free(V);free(w);free(alpha);free(beta);free(dots);
    free(v0);free(w_buf);free(z2_buf);free(mv_buf);
    free(Gmat);free(g_c3);

    printf("Total time: %.0fs (%.1f min)\n",omp_get_wtime()-T0,(omp_get_wtime()-T0)/60.0);fflush(stdout);
    return nullity;
}

/* Main */
int main(int argc,char*argv[]){
    const char*dir=(argc>1)?argv[1]:".";
    int k_iters=(argc>2)?atoi(argv[2]):20;
    int nt=omp_get_max_threads();
    if(nt>15)nt=15;
    omp_set_num_threads(nt);
    double T0=omp_get_wtime();

    printf("=== sl_3 Projected Lanczos (pure C, AVX-512, OpenMP) ===\nThreads=%d\n\n",nt);fflush(stdout);

    load_algebra(dir);build_weight_classes();
    printf("mult_bar:%d delta_bar:%d\n\n",NNZ_MB,NNZ_DB);fflush(stdout);

    /* Build C¹, C², d1 */
    printf("[A] Building C1, C2, d1...\n");fflush(stdout);
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

    /* Build d1 (corrected) */
    printf("[B] Building d1...\n");fflush(stdout);
    t=omp_get_wtime();
    spmat*d1=spmat_new(n_c2,n_c1,10000000);
    for(int ci=0;ci<n_c1;ci++){
        int hj=c1j[ci],hk=c1k[ci];
        for(int i=0;i<n_mb_by_b[hj];i++){int idx=mb_by_b[hj][i];
            int32_t r=hash_insert(fkey(mult_bar[idx].a,hk,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
        for(int i=0;i<n_mb_by_l[hk];i++){int idx=mb_by_l[hk][i];
            int32_t r=hash_insert(fkey(mult_bar[idx].a,mult_bar[idx].b,hj));
            if(r>=0)spmat_add(d1,r,ci,-mb_val(idx));}
        for(int i=0;i<n_mb_by_a[hj];i++){int idx=mb_by_a[hj][i];
            int32_t r=hash_insert(fkey(hk,mult_bar[idx].b,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
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

    g_n_c1=n_c1;g_n_c2=n_c2;g_d1=d1;g_c2_tuple=c2_tuple;g_c2_type=c2_type;

    /* Run Projected Lanczos */
    printf("[C] Starting Projected Lanczos...\n\n");fflush(stdout);
    int nullity=solve_projected_lanczos(k_iters,nt);

    printf("\n=== FINAL RESULT ===\n");
    printf("  nullity per shift = %d\n",nullity);
    printf("  dim H̃²_b(B⁺) = %d\n",nullity);
    printf("  dim HH²(u_q(sl_3)) = 9 - %d = %d\n",nullity,9-nullity);
    printf("\nTotal: %.1fs (%.1f min)\n",omp_get_wtime()-T0,(omp_get_wtime()-T0)/60.0);fflush(stdout);

    free(d1->rows);free(d1->cols);free(d1->vals);free(d1);
    free(c2hash);
    return 0;
}
