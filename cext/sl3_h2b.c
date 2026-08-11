/*
 * sl3_h2b.c — Pure C computation of dim H̃²_b(B⁺(u_q(sl_3))) at ℓ=3.
 * AVX-512 + OpenMP.  No Python.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o sl3_h2b sl3_h2b.c \
 *       -lm -L/usr/lib64 -lopenblas -llapack -lgfortran
 *
 * Run:
 *   ./sl3_h2b [data_dir]
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

#define ELL 3
#define DIM 243
#define NNZ_MB 27671
#define NNZ_DB 2647

extern void zheevd_(const char*,const char*,const int*,dcomplex*,const int*,double*,dcomplex*,const int*,double*,const int*,int*,const int*,int*);
extern void zheev_(const char*,const char*,const int*,dcomplex*,const int*,double*,dcomplex*,const int*,double*,int*);
extern void zgemm_(const char*,const char*,const int*,const int*,const int*,const dcomplex*,const dcomplex*,const int*,const dcomplex*,const int*,const dcomplex*,dcomplex*,const int*);
extern void zgeqrf_(const int*,const int*,dcomplex*,const int*,dcomplex*,dcomplex*,const int*,int*);
extern void zungqr_(const int*,const int*,const int*,dcomplex*,const int*,const dcomplex*,dcomplex*,const int*,int*);

/* ---- Algebra data (packed to match numpy export) ---- */
static int32_t weights[DIM][2];
struct __attribute__((packed)) mb_entry { int32_t l,a,b; double re,im; };
struct __attribute__((packed)) db_entry { int32_t c,j,k; double re,im; };
static struct mb_entry mult_bar[NNZ_MB];
static struct db_entry delta_bar[NNZ_DB];

static dcomplex mb_val(int i){return mult_bar[i].re+I*mult_bar[i].im;}
static dcomplex db_val(int i){return delta_bar[i].re+I*delta_bar[i].im;}

/* Inverse tables */
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
    /* Build inverse tables */
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

/* Weight classes */
static int32_t w2i_elems[ELL*ELL][81];
static int w2i_n[ELL*ELL];
static int wt_key(int w0,int w1){return w0*ELL+w1;}
/* Weight class iteration order (matching Python dict insertion order = PBW index order) */
static int w2i_order[ELL*ELL]; /* keys in insertion order */
static int w2i_nkeys;
static void build_weight_classes(void){
    for(int i=1;i<DIM;i++){
        int k=wt_key(weights[i][0],weights[i][1]);
        if(w2i_n[k]==0) w2i_order[w2i_nkeys++]=k; /* first time seeing this key */
        w2i_elems[k][w2i_n[k]++]=i;
    }
}

/* ---- Unified C² hash: f-rows and g-rows in ONE table ---- */
/* Key: bit 63 = type (0=f, 1=g), bits 47-32 = first, 31-16 = second, 15-0 = third */
#define HASH_SIZE (1<<26)
#define C2_MAX 20000000
typedef struct{uint64_t key;int32_t idx;}hentry;
static hentry *c2hash;
static int32_t c2count;
static int32_t *c2_tuple; /* [3*C2_MAX] */
static int8_t *c2_type;   /* [C2_MAX] */

static uint64_t fkey(int a,int b,int c){return ((uint64_t)a<<33)|((uint64_t)b<<17)|((uint64_t)c<<1)|0;}
static uint64_t gkey(int c,int j,int k){return ((uint64_t)c<<33)|((uint64_t)j<<17)|((uint64_t)k<<1)|1;}

static void hash_init(void){
    c2hash=calloc(HASH_SIZE,sizeof(hentry));
    for(int i=0;i<HASH_SIZE;i++)c2hash[i].idx=-1;
    c2count=0;
    c2_tuple=malloc(C2_MAX*3*sizeof(int32_t));
    c2_type=malloc(C2_MAX*sizeof(int8_t));
    if(!c2_tuple||!c2_type){fprintf(stderr,"malloc c2_tuple/c2_type failed\n");exit(1);}
}
static int32_t hash_insert(uint64_t key){
    uint64_t h=key%HASH_SIZE;
    while(c2hash[h].idx!=-1){if(c2hash[h].key==key)return c2hash[h].idx;h=(h+1)%HASH_SIZE;}
    if(c2count>=C2_MAX){fprintf(stderr,"FATAL: c2_count=%d >= C2_MAX=%d\n",c2count,C2_MAX);exit(1);}
    c2hash[h].key=key;c2hash[h].idx=c2count++;return c2hash[h].idx;
}
static int32_t hash_lookup(uint64_t key){
    uint64_t h=key%HASH_SIZE;
    while(c2hash[h].idx!=-1){if(c2hash[h].key==key)return c2hash[h].idx;h=(h+1)%HASH_SIZE;}
    return -1;
}

/* ---- Sparse matrix ---- */
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

/* ---- LAPACK helpers ---- */
static void ortho(dcomplex*Q,int k,int m){
    int n=k,lda=m,info;dcomplex*tau=malloc(k*16);
    int lwork=-1;dcomplex wk;zgeqrf_(&m,&n,Q,&lda,tau,&wk,&lwork,&info);
    lwork=(int)creal(wk);dcomplex*work=malloc(lwork*16);
    zgeqrf_(&m,&n,Q,&lda,tau,work,&lwork,&info);
    zungqr_(&m,&n,&n,Q,&lda,tau,work,&lwork,&info);
    free(work);free(tau);
}
static void ritz_values(dcomplex*Q,dcomplex*W,int k,int m,dcomplex*Y,
    int64_t*d2_rows,int32_t*d2_cols,dcomplex*d2_vals,int64_t d2_nnz,
    double*evals){
    /* W = A*A Q column by column (using d2 as the operator) */
    for(int j=0;j<k;j++){
        memset(Y,0,m*sizeof(dcomplex));
        for(int64_t e=0;e<d2_nnz;e++)Y[d2_rows[e]]+=d2_vals[e]*Q[j*m+d2_cols[e]];
        memset(W+j*m,0,m*sizeof(dcomplex));
        for(int64_t e=0;e<d2_nnz;e++)W[j*m+d2_cols[e]]+=conj(d2_vals[e])*Y[d2_rows[e]];
    }
    dcomplex*G=malloc(k*k*16);dcomplex alpha=1,beta=0;int kk=m;
    zgemm_("C","N",&k,&k,&kk,&alpha,Q,&m,W,&m,&beta,G,&k);
    int lwork=4*k*k,lrwork=3*k*k,liwork=7*k,info;
    dcomplex*work=malloc(lwork*16);double*rwork=malloc(lrwork*8);int*iwork=malloc(liwork*4);
    zheevd_("N","U",&k,G,&k,evals,work,&lwork,rwork,&lrwork,iwork,&liwork,&info);
    free(work);free(rwork);free(iwork);free(G);
}

int main(int argc,char*argv[]){
    const char*dir=(argc>1)?argv[1]:".";
    int nt=omp_get_max_threads();double T0=omp_get_wtime();

    printf("=== sl_3 H2b (pure C, AVX-512, OpenMP) ===\nThreads=%d\n\n",nt);fflush(stdout);

    printf("[1] Loading algebra...\n");fflush(stdout);
    load_algebra(dir);build_weight_classes();
    printf("  mult_bar:%d delta_bar:%d\n",NNZ_MB,NNZ_DB);
    for(int k=0;k<ELL*ELL;k++)if(w2i_n[k])printf("  wt(%d,%d):%d\n",k/ELL,k%ELL,w2i_n[k]);
    fflush(stdout);

    /* ---- C1, C2, C3 enumeration for shift (0,0) ---- */
    printf("\n[2] Building C1, C2 for shift(0,0)...\n");fflush(stdout);
    int s0=0,s1=0;

    /* C1: (j,k) with wt(j)-wt(k)=(0,0) */
    int32_t n_c1=0;int32_t c1j[20000],c1k[20000];
    for(int wi=0;wi<w2i_nkeys;wi++){int wj=w2i_order[wi];
    for(int wj2=0;wj2<w2i_nkeys;wj2++){int wk=w2i_order[wj2];
        if((wj/ELL-wk/ELL)%ELL==s0&&(wj%ELL-wk%ELL)%ELL==s1)
            for(int ji=0;ji<w2i_n[wj];ji++)for(int ki=0;ki<w2i_n[wk];ki++)
                {c1j[n_c1]=w2i_elems[wj][ji];c1k[n_c1]=w2i_elems[wk][ki];n_c1++;}
    }}

    /* C2: unified hash for f-rows and g-rows */
    hash_init();
    int32_t n_f=0,n_g=0;
    /* f-rows: (a,b,l) with wt(l)-wt(a)-wt(b)=(0,0) */
    printf("  Building f-rows...\n");fflush(stdout);
    for(int wai=0;wai<w2i_nkeys;wai++){int wa=w2i_order[wai];
        for(int wbi=0;wbi<w2i_nkeys;wbi++){int wb=w2i_order[wbi];
            int wl=wt_key((s0+wa/ELL+wb/ELL)%ELL,(s1+wa%ELL+wb%ELL)%ELL);
            if(!w2i_n[wl])continue;
            for(int ai=0;ai<w2i_n[wa];ai++)for(int bi=0;bi<w2i_n[wb];bi++)for(int li=0;li<w2i_n[wl];li++){
                int a=w2i_elems[wa][ai],b=w2i_elems[wb][bi],l=w2i_elems[wl][li];
                int32_t idx=hash_insert(fkey(a,b,l));
                c2_tuple[idx*3]=a;c2_tuple[idx*3+1]=b;c2_tuple[idx*3+2]=l;c2_type[idx]=0;
                n_f++;
            }
        }
    }
    printf("  f-rows: %d (count=%d)\n",n_f,c2count);fflush(stdout);
    /* g-rows: (c,j,k) with wt(j)+wt(k)-wt(c)=(0,0) */
    printf("  Building g-rows...\n");fflush(stdout);
    for(int wci=0;wci<w2i_nkeys;wci++){int wc=w2i_order[wci];
        for(int wji=0;wji<w2i_nkeys;wji++){int wj=w2i_order[wji];
            int wk0=((s0-wj/ELL+wc/ELL)%ELL+ELL)%ELL;
            int wk1=((s1-wj%ELL+wc%ELL)%ELL+ELL)%ELL;
            int wk=wt_key(wk0,wk1);
            if(!w2i_n[wk])continue;
            for(int ci=0;ci<w2i_n[wc];ci++)for(int ji=0;ji<w2i_n[wj];ji++)for(int ki=0;ki<w2i_n[wk];ki++){
                int c=w2i_elems[wc][ci],j=w2i_elems[wj][ji],k=w2i_elems[wk][ki];
                int32_t idx=hash_insert(gkey(c,j,k));
                c2_tuple[idx*3]=c;c2_tuple[idx*3+1]=j;c2_tuple[idx*3+2]=k;c2_type[idx]=1;
                n_g++;
            }
        }
    }
    printf("  g-rows: %d (count=%d)\n",n_g,c2count);fflush(stdout);
    int32_t n_c2=c2count;
    printf("  C1:%d C2:%d (%df+%dg)\n",n_c1,n_c2,n_f,n_g);fflush(stdout);

    /* ---- Build d1 ---- */
    printf("\n[3] Building d1...\n");fflush(stdout);
    double t=omp_get_wtime();
    spmat*d1=spmat_new(n_c2,n_c1,10000000);

    for(int ci=0;ci<n_c1;ci++){
        int hj=c1j[ci],hk=c1k[ci];
        /* ∂^h: 3 terms → f-rows */
        /* T1: a·h(b) where b=hk,h(hk)=hj. mult_bar[l,a,hj] */
        for(int i=0;i<n_mb_by_b[hj];i++){int idx=mb_by_b[hj][i];
            int32_t r=hash_lookup(fkey(mult_bar[idx].a,hk,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
        /* T2: -h(a·b) where h(x)=hj if x=hk. mult_bar[hk,a,b] */
        for(int i=0;i<n_mb_by_l[hk];i++){int idx=mb_by_l[hk][i];
            int32_t r=hash_lookup(fkey(mult_bar[idx].a,mult_bar[idx].b,hj));
            if(r>=0)spmat_add(d1,r,ci,-mb_val(idx));}
        /* T3: h(a)·b where a=hk. mult_bar[l,hj,b] */
        for(int i=0;i<n_mb_by_a[hj];i++){int idx=mb_by_a[hj][i];
            int32_t r=hash_lookup(fkey(hk,mult_bar[idx].b,mult_bar[idx].l));
            if(r>=0)spmat_add(d1,r,ci,mb_val(idx));}
        /* ∂^c: 3 terms → g-rows, OVERALL MINUS: ∂_b = (∂^h, -∂^c) */
        /* T1: c₁⊗h(c₂) — negate: -db_val */
        for(int i=0;i<n_db_by_k[hk];i++){int idx=db_by_k[hk][i];
            int32_t r=hash_lookup(gkey(delta_bar[idx].c,delta_bar[idx].j,hj));
            if(r>=0)spmat_add(d1,r,ci,-db_val(idx));}
        /* T2: -Δ(h(c)) — double negate: +db_val */
        for(int i=0;i<n_db_by_c[hj];i++){int idx=db_by_c[hj][i];
            int32_t r=hash_lookup(gkey(hk,delta_bar[idx].j,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,db_val(idx));}
        /* T3: h(c₁)⊗c₂ — negate: -db_val */
        for(int i=0;i<n_db_by_j[hk];i++){int idx=db_by_j[hk][i];
            int32_t r=hash_lookup(gkey(delta_bar[idx].c,hj,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,-db_val(idx));}
    }
    printf("  d1: %lld nnz (%.1fs)\n",(long long)d1->nnz,omp_get_wtime()-t);fflush(stdout);

    /* ---- Compute rank(d1) via DENSE Gram matrix + LAPACK zheevd ---- */
    printf("\n[4] Computing rank(d1) via dense Gram matrix...\n");fflush(stdout);
    t=omp_get_wtime();

    /* Check d1 entries for out-of-bounds indices */
    for(int64_t e=0;e<d1->nnz;e++){
        if(d1->rows[e]<0||d1->rows[e]>=n_c2||d1->cols[e]<0||d1->cols[e]>=n_c1){
            fprintf(stderr,"FATAL: d1 entry %lld: row=%d (max %d), col=%d (max %d)\n",
                (long long)e,d1->rows[e],n_c2-1,d1->cols[e],n_c1-1);
            exit(1);
        }
    }
    printf("  d1 bounds check passed\n");fflush(stdout);

    /* G = d1* @ d1 (dense, n_c1 × n_c1, complex128) */
    /* G[i,j] = sum_e conj(d1[e,i]) * d1[e,j]  where e iterates over rows */
    /* Build by grouping d1 entries by row, then computing outer products */
    int32_t ng = n_c1; /* 19522 — Gram matrix dimension */
    printf("  Allocating G: %d x %d = %.2f GB\n", ng, ng, (double)ng*ng*16/1e9);
    fflush(stdout);

    dcomplex *Gmat = calloc((size_t)ng * ng, sizeof(dcomplex));
    if(!Gmat){fprintf(stderr,"calloc G failed (%.2f GB)\n",(double)ng*ng*16/1e9);exit(1);}

    /* Sort d1 entries by row (counting sort) */
    int32_t *row_cnt = calloc(n_c2, sizeof(int32_t));
    for(int64_t e=0;e<d1->nnz;e++) row_cnt[d1->rows[e]]++;
    int32_t *row_off = calloc(n_c2, sizeof(int32_t));
    int32_t acc2=0;
    for(int r=0;r<n_c2;r++){row_off[r]=acc2;acc2+=row_cnt[r];}
    int32_t *sorted_cols = malloc(d1->nnz*sizeof(int32_t));
    dcomplex *sorted_vals = malloc(d1->nnz*sizeof(dcomplex));
    int32_t *pos = calloc(n_c2, sizeof(int32_t));
    for(int64_t e=0;e<d1->nnz;e++){
        int32_t r=d1->rows[e];
        int32_t p=row_off[r]+pos[r]++;
        sorted_cols[p]=d1->cols[e];
        sorted_vals[p]=d1->vals[e];
    }

    /* For each row r, compute outer product of its entries */
    /* G[ci,cj] += conj(val_ci) * val_cj  for all ci,cj in the same row */
    /* Single-threaded — only 0.2s, avoids OpenMP atomic issues with complex */
    for(int32_t r=0;r<n_c2;r++){
        int32_t start=row_off[r];
        int32_t cnt=row_cnt[r];
        if(cnt==0) continue;
        for(int32_t i=0;i<cnt;i++){
            int32_t ci=sorted_cols[start+i];
            dcomplex vi=sorted_vals[start+i];
            for(int32_t j=0;j<cnt;j++){
                int32_t cj=sorted_cols[start+j];
                dcomplex vj=sorted_vals[start+j];
                Gmat[ci + (int64_t)cj*ng] += conj(vi)*vj;
            }
        }
    }

    free(row_cnt);free(row_off);free(sorted_cols);free(sorted_vals);free(pos);
    printf("  Gram matrix built (%.1fs)\n",omp_get_wtime()-t);fflush(stdout);

    /* Eigenvalues of Hermitian G via LAPACK zheevd (divide-and-conquer, FAST) */
    printf("  Computing eigenvalues via zheevd...\n");fflush(stdout);
    double t2=omp_get_wtime();
    double *evals_all = malloc(ng*sizeof(double));
    int lwork=-1,lrwork=-1,liwork=-1,info;
    dcomplex wkopt;
    double rwkopt=0;
    int iwkopt=0;
    printf("  zheevd workspace query...\n");fflush(stdout);
    zheevd_("N","U",&ng,Gmat,&ng,evals_all,&wkopt,&lwork,&rwkopt,&lrwork,&iwkopt,&liwork,&info);
    printf("  query: info=%d, lwork=%d, lrwork=%d, liwork=%d\n",info,(int)creal(wkopt),(int)rwkopt,iwkopt);fflush(stdout);
    if(info!=0){fprintf(stderr,"zheevd query failed: info=%d\n",info);exit(1);}
    lwork=(int)creal(wkopt);
    lrwork=(int)rwkopt;
    liwork=iwkopt;
    printf("  lwork=%d (%.2f GB), lrwork=%d (%.2f GB), liwork=%d\n",
           lwork,lwork*16/1e9,lrwork,lrwork*8/1e9,liwork);fflush(stdout);
    dcomplex *work=malloc((size_t)lwork*sizeof(dcomplex));
    double *rwork=malloc((size_t)lrwork*sizeof(double));
    int *iwork=malloc((size_t)liwork*sizeof(int));
    if(!work||!rwork||!iwork){fprintf(stderr,"malloc work/rwork/iwork failed\n");exit(1);}
    printf("  calling zheevd...\n");fflush(stdout);
    zheevd_("N","U",&ng,Gmat,&ng,evals_all,work,&lwork,rwork,&lrwork,iwork,&liwork,&info);
    printf("  zheevd returned: info=%d\n",info);fflush(stdout);
    if(info!=0){fprintf(stderr,"zheevd failed: info=%d\n",info);exit(1);}
    free(work);free(rwork);free(iwork);free(Gmat);
    printf("  zheevd done (%.1fs)\n",omp_get_wtime()-t2);fflush(stdout);

    /* Count near-zero eigenvalues */
    double lam_max = evals_all[ng-1];
    double tol_d1 = lam_max * 1e-10;
    int null_d1 = 0;
    for(int i=0;i<ng;i++) if(evals_all[i] < tol_d1) null_d1++;
    int rank_d1 = ng - null_d1;

    printf("  eigenvalue range: [%.4e, %.4e]\n", evals_all[0], lam_max);
    printf("  smallest 10: ");
    for(int i=0;i<10 && i<ng;i++) printf("%.4e ", evals_all[i]);
    printf("\n");
    printf("  nullity(d1) = %d, rank(d1) = %d (expect 2, 19520)\n", null_d1, rank_d1);
    printf("  (%.1fs)\n",omp_get_wtime()-t);fflush(stdout);

    /* ---- Build d2 and verify d2∘d1=0 ---- */
    /* Formula: ∂_b(f,g) = (∂ʰf, ∂ʰg + ∂ᶜf, -∂ᶜg)
       NOTE: sign trick gives +∂ᶜf for p=2 (B^{2,0}), NOT -∂ᶜf.
       The Python v5 code had -∂ᶜf which is WRONG. */
    printf("\n[5] Verifying d2∘d1=0 via C³ hash accumulation...\n");fflush(stdout);
    t=omp_get_wtime();

    /* C³ hash: 3 components packed into 64-bit key.
       h-rows (comp=0): (a,b,c,t) — 4 indices, but t≤242, a,b,c≤242
         key = (0<<60) | (a<<48) | (b<<32) | (c<<16) | t   [but a,b,c can be up to 242 < 256, fits in 16 bits]
       m-rows (comp=1): (a,b,j,k) — key = (1<<60) | (a<<48) | (b<<32) | (j<<16) | k
       c-rows (comp=2): (c,j,k,l) — key = (2<<60) | (c<<48) | (j<<32) | (k<<16) | l
       All indices < 256 (DIM=243), so 8 bits each, packed in 32 bits + 3-bit comp. */
    #define C3_HASH_SIZE (1u<<27)
    #define C3_MAX 200000000
    typedef struct{uint64_t key;dcomplex val;}c3entry;
    static c3entry *c3hash;
    static int32_t c3count;
    c3hash=malloc(C3_HASH_SIZE*sizeof(c3entry));
    if(!c3hash){fprintf(stderr,"malloc c3hash failed\n");exit(1);}
    for(uint32_t i=0;i<C3_HASH_SIZE;i++)c3hash[i].key=0xFFFFFFFFFFFFFFFFULL;
    /* key=0 is valid (comp=0, all zeros), so use 0xFFFF... as sentinel */

    int n_tests=3;double max_ratio=0;
    srand(791);
    for(int test=0;test<n_tests;test++){
        /* Reset hash */
        for(uint32_t i=0;i<C3_HASH_SIZE;i++)c3hash[i].key=0xFFFFFFFFFFFFFFFFULL;
        c3count=0;

        /* Random h ∈ C¹ */
        dcomplex*hv=calloc(n_c1,16);
        for(int i=0;i<n_c1;i++)hv[i]=(rand()/(double)RAND_MAX-.5)+I*(rand()/(double)RAND_MAX-.5);

        /* v = d1 @ h (dense, n_c2) */
        dcomplex*vv=calloc(n_c2,16);
        for(int64_t e=0;e<d1->nnz;e++)vv[d1->rows[e]]+=d1->vals[e]*hv[d1->cols[e]];
        double nv2=0;for(int i=0;i<n_c2;i++)nv2+=creal(conj(vv[i])*vv[i]);

        /* Apply d2 to vv: for each C² entry i, accumulate into C³ hash.
           f-type (af,bf,lf), val=vv[i]:
             ∂ʰf (4 terms → h-rows): signs +,-,+,-
               T1: mult_bar[t,a,lf] → h-row(a, af, bf, t), val=+v*vv[i]
               T2: mult_bar[af,a,b] → h-row(a, b, bf, lf), val=-v*vv[i]
               T3: mult_bar[bf,b,c] → h-row(af, b, c, lf), val=+v*vv[i]
               T4: mult_bar[t,lf,c] → h-row(af, bf, c, t), val=-v*vv[i]
             +∂ᶜf (3 terms → m-rows): signs -, +, -
               [Sign formula: (-1)^(p+1) for left coaction, (-1)^p for Δ on output, (-1)^(p+q+1) for right coaction, p=2]
               T1: delta_bar[a,j,af] → m-row(a, bf, j, lf), val=-v*vv[i]  [a₁⊗f(a₂,b), left coaction]
               T2: delta_bar[lf,j,k] → m-row(af, bf, j, k), val=+v*vv[i]  [Δ(f(a,b)), output Δ]
               T3: delta_bar[b,bf,k] → m-row(af, b, lf, k), val=-v*vv[i]  [f(a,b₁)⊗b₂, right coaction]
           g-type (cf,jf,kf), val=vv[i]:
             ∂ʰg (3 terms → m-rows): signs +,-,+
               T1: mult_bar[j,a,jf] → m-row(a, cf, j, kf), val=+v*vv[i]  [a·g(b)]
               T2: mult_bar[cf,a,b] → m-row(a, b, jf, kf), val=-v*vv[i]  [g(ab)]
               T3: mult_bar[k,b,kf] → m-row(cf, b, jf, k), val=+v*vv[i]  [g(a)·b]
             -∂ᶜg (4 terms → c-rows): signs -, +, -, +  [p=1, q=1 → 2 coactions + 2 Δ-on-outputs]
               T1: delta_bar[c,j,cf] → c-row(c, j, jf, kf), val=-v*vv[i]  [a₁⊗g(a₂), left coaction]
               T2: delta_bar[jf,j,K] → c-row(cf, j, K, kf), val=+v*vv[i]  [(Δ⊗id)g(a), Δ on 1st output]
               T3: delta_bar[kf,j,K] → c-row(cf, jf, j, K), val=-v*vv[i]  [(id⊗Δ)g(a), Δ on 2nd output]
               T4: delta_bar[c,cf,l] → c-row(c, jf, kf, l), val=+v*vv[i]  [g(a₁)⊗a₂, right coaction]
        */
        for(int32_t i=0;i<n_c2;i++){
            dcomplex vi=vv[i];
            if(cabs(vi)<1e-15)continue;
            if(c2_type[i]==0){
                /* f-type */
                int af=c2_tuple[i*3],bf=c2_tuple[i*3+1],lf=c2_tuple[i*3+2];
                /* ∂ʰf T1: mult_bar[t,a,lf] → h-row(a,af,bf,t), +v */
                for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];
                    uint64_t key=((uint64_t)0<<60)|((uint64_t)mult_bar[idx].a<<48)|((uint64_t)af<<32)|((uint64_t)bf<<16)|((uint64_t)mult_bar[idx].l);
                    dcomplex contr=mb_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* ∂ʰf T2: mult_bar[af,a,b] → h-row(a,b,bf,lf), -v */
                for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];
                    uint64_t key=((uint64_t)0<<60)|((uint64_t)mult_bar[idx].a<<48)|((uint64_t)mult_bar[idx].b<<32)|((uint64_t)bf<<16)|((uint64_t)lf);
                    dcomplex contr=-mb_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* ∂ʰf T3: mult_bar[bf,b,c] → h-row(af,b,c,lf), +v */
                for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];
                    uint64_t key=((uint64_t)0<<60)|((uint64_t)af<<48)|((uint64_t)mult_bar[idx].a<<32)|((uint64_t)mult_bar[idx].b<<16)|((uint64_t)lf);
                    dcomplex contr=mb_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* ∂ʰf T4: mult_bar[t,lf,c] → h-row(af,bf,c,t), -v */
                for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];
                    uint64_t key=((uint64_t)0<<60)|((uint64_t)af<<48)|((uint64_t)bf<<32)|((uint64_t)mult_bar[idx].b<<16)|((uint64_t)mult_bar[idx].l);
                    dcomplex contr=-mb_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* +∂ᶜf T1: delta_bar[a,j,af] → m-row(a,bf,j,lf), -v  [left coaction, sign (-1)^(p+1)=-1 for p=2] */
                for(int e=0;e<n_db_by_k[af];e++){int idx=db_by_k[af][e];
                    uint64_t key=((uint64_t)1<<60)|((uint64_t)delta_bar[idx].c<<48)|((uint64_t)bf<<32)|((uint64_t)delta_bar[idx].j<<16)|((uint64_t)lf);
                    dcomplex contr=-db_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* +∂ᶜf T2: delta_bar[lf,j,k] → m-row(af,bf,j,k), +v  [output Δ, sign (-1)^p=+1 for p=2] */
                for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];
                    uint64_t key=((uint64_t)1<<60)|((uint64_t)af<<48)|((uint64_t)bf<<32)|((uint64_t)delta_bar[idx].j<<16)|((uint64_t)delta_bar[idx].k);
                    dcomplex contr=db_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* +∂ᶜf T3: delta_bar[b,bf,k] → m-row(af,b,lf,k), -v  [right coaction, sign (-1)^(p+q+1)=-1 for p=2,q=0] */
                for(int e=0;e<n_db_by_j[bf];e++){int idx=db_by_j[bf][e];
                    uint64_t key=((uint64_t)1<<60)|((uint64_t)af<<48)|((uint64_t)delta_bar[idx].c<<32)|((uint64_t)lf<<16)|((uint64_t)delta_bar[idx].k);
                    dcomplex contr=-db_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
            } else {
                /* g-type */
                int cf=c2_tuple[i*3],jf=c2_tuple[i*3+1],kf=c2_tuple[i*3+2];
                /* ∂ʰg T1: mult_bar[j,a,jf] → m-row(a,cf,j,kf), +v */
                for(int e=0;e<n_mb_by_b[jf];e++){int idx=mb_by_b[jf][e];
                    uint64_t key=((uint64_t)1<<60)|((uint64_t)mult_bar[idx].a<<48)|((uint64_t)cf<<32)|((uint64_t)mult_bar[idx].l<<16)|((uint64_t)kf);
                    dcomplex contr=mb_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* ∂ʰg T2: mult_bar[cf,a,b] → m-row(a,b,jf,kf), -v */
                for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];
                    uint64_t key=((uint64_t)1<<60)|((uint64_t)mult_bar[idx].a<<48)|((uint64_t)mult_bar[idx].b<<32)|((uint64_t)jf<<16)|((uint64_t)kf);
                    dcomplex contr=-mb_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* ∂ʰg T3: mult_bar[k,b,kf] → m-row(cf,b,jf,k), +v */
                for(int e=0;e<n_mb_by_b[kf];e++){int idx=mb_by_b[kf][e];
                    uint64_t key=((uint64_t)1<<60)|((uint64_t)cf<<48)|((uint64_t)mult_bar[idx].a<<32)|((uint64_t)jf<<16)|((uint64_t)mult_bar[idx].l);
                    dcomplex contr=mb_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* -∂ᶜg T1: delta_bar[c,j,cf] → c-row(c,j,jf,kf), -v  [left coaction, sign -(-1)^(p+1)=-1 for p=1] */
                for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];
                    uint64_t key=((uint64_t)2<<60)|((uint64_t)delta_bar[idx].c<<48)|((uint64_t)delta_bar[idx].j<<32)|((uint64_t)jf<<16)|((uint64_t)kf);
                    dcomplex contr=-db_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* -∂ᶜg T2: delta_bar[jf,j,K] → c-row(cf,j,K,kf), +v  [Δ on 1st output, sign -(-1)^p=+1 for p=1] */
                for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];
                    uint64_t key=((uint64_t)2<<60)|((uint64_t)cf<<48)|((uint64_t)delta_bar[idx].j<<32)|((uint64_t)delta_bar[idx].k<<16)|((uint64_t)kf);
                    dcomplex contr=db_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* -∂ᶜg T3: delta_bar[kf,j,K] → c-row(cf,jf,j,K), -v  [Δ on 2nd output, sign -(-1)^(p+1)=-1 for p=1] */
                for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];
                    uint64_t key=((uint64_t)2<<60)|((uint64_t)cf<<48)|((uint64_t)jf<<32)|((uint64_t)delta_bar[idx].j<<16)|((uint64_t)delta_bar[idx].k);
                    dcomplex contr=-db_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
                /* -∂ᶜg T4: delta_bar[c,cf,l] → c-row(c,jf,kf,l), +v  [right coaction, sign -(-1)^(p+q+1)=+1 for p=1,q=1] */
                for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];
                    uint64_t key=((uint64_t)2<<60)|((uint64_t)delta_bar[idx].c<<48)|((uint64_t)jf<<32)|((uint64_t)kf<<16)|((uint64_t)delta_bar[idx].k);
                    dcomplex contr=db_val(idx)*vi;
                    uint32_t h=key%C3_HASH_SIZE;
                    while(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL&&c3hash[h].key!=key)h=(h+1)%C3_HASH_SIZE;
                    if(c3hash[h].key==0xFFFFFFFFFFFFFFFFULL){c3hash[h].key=key;c3hash[h].val=contr;c3count++;}
                    else c3hash[h].val+=contr;
                }
            }
        }

        /* Compute ||w||^2 from C³ hash */
        double nw2=0;
        for(uint32_t h=0;h<C3_HASH_SIZE;h++){
            if(c3hash[h].key!=0xFFFFFFFFFFFFFFFFULL)nw2+=creal(conj(c3hash[h].val)*c3hash[h].val);
        }
        double ratio = (nv2>1e-10) ? sqrt(nw2/nv2) : 0;
        if(ratio>max_ratio)max_ratio=ratio;
        printf("  test %d: ||d1(h)||=%.4e ||d2(d1(h))||=%.4e ratio=%.4e  [C³ nnz=%d]\n",
               test,sqrt(nv2),sqrt(nw2),ratio,c3count);fflush(stdout);

        free(hv);free(vv);
    }
    free(c3hash);

    printf("\n  max ||d2∘d1||/||d1|| = %.4e %s\n",max_ratio,
           max_ratio<1e-8?"PASS ✓":"FAIL ✗");fflush(stdout);
    printf("  (%.1fs)\n",omp_get_wtime()-t);fflush(stdout);

    /* ---- Summary ---- */
    printf("\n=== SUMMARY ===\n");
    printf("  d1: %d × %d, nnz=%lld\n",d1->n_rows,d1->n_cols,(long long)d1->nnz);
    printf("  nullity(d1)~%d rank(d1)~%d\n",null_d1,n_c1-null_d1);
    printf("\nTotal: %.1fs\n",omp_get_wtime()-T0);fflush(stdout);

    spmat_free(d1);
    free(c2hash);
    return 0;
}
