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
static void build_weight_classes(void){
    for(int i=1;i<DIM;i++){int k=wt_key(weights[i][0],weights[i][1]);w2i_elems[k][w2i_n[k]++]=i;}
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
    for(int wj=0;wj<ELL*ELL;wj++)for(int wk=0;wk<ELL*ELL;wk++)
        if((wj/ELL-wk/ELL)%ELL==s0&&(wj%ELL-wk%ELL)%ELL==s1)
            for(int ji=0;ji<w2i_n[wj];ji++)for(int ki=0;ki<w2i_n[wk];ki++)
                {c1j[n_c1]=w2i_elems[wj][ji];c1k[n_c1]=w2i_elems[wk][ki];n_c1++;}

    /* C2: unified hash for f-rows and g-rows */
    hash_init();
    int32_t n_f=0,n_g=0;
    /* f-rows: (a,b,l) with wt(l)-wt(a)-wt(b)=(0,0) */
    printf("  Building f-rows...\n");fflush(stdout);
    for(int wa=0;wa<ELL*ELL;wa++){if(!w2i_n[wa])continue;
        for(int wb=0;wb<ELL*ELL;wb++){if(!w2i_n[wb])continue;
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
    for(int wc=0;wc<ELL*ELL;wc++){if(!w2i_n[wc])continue;
        for(int wj=0;wj<ELL*ELL;wj++){if(!w2i_n[wj])continue;
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

    /* ---- Compute rank(d1) via subspace iteration ---- */
    printf("\n[4] Computing rank(d1)...\n");fflush(stdout);
    t=omp_get_wtime();
    /* Power iteration */
    dcomplex*h=calloc(n_c1,16),*v=calloc(n_c2,16),*r=calloc(n_c1,16);
    if(!h||!v||!r){fprintf(stderr,"calloc failed for power iteration vectors\n");exit(1);}
    srand(123);for(int i=0;i<n_c1;i++)h[i]=(rand()/(double)RAND_MAX-.5)+I*(rand()/(double)RAND_MAX-.5);
    double lam=0;
    /* Check d1 entries for out-of-bounds indices */
    for(int64_t e=0;e<d1->nnz;e++){
        if(d1->rows[e]<0||d1->rows[e]>=n_c2||d1->cols[e]<0||d1->cols[e]>=n_c1){
            fprintf(stderr,"FATAL: d1 entry %lld: row=%d (max %d), col=%d (max %d)\n",
                (long long)e,d1->rows[e],n_c2-1,d1->cols[e],n_c1-1);
            exit(1);
        }
    }
    printf("  d1 bounds check passed\n");fflush(stdout);
    for(int it=0;it<10;it++){
        memset(v,0,n_c2*16);
        for(int64_t e=0;e<d1->nnz;e++)v[d1->rows[e]]+=d1->vals[e]*h[d1->cols[e]];
        double av2=0;for(int i=0;i<n_c2;i++)av2+=creal(conj(v[i])*v[i]);lam=av2;
        memset(r,0,n_c1*16);
        for(int64_t e=0;e<d1->nnz;e++)r[d1->cols[e]]+=conj(d1->vals[e])*v[d1->rows[e]];
        double r2=0;for(int i=0;i<n_c1;i++)r2+=creal(conj(r[i])*r[i]);
        dcomplex inv=1.0/sqrt(r2);for(int i=0;i<n_c1;i++)h[i]=r[i]*inv;
        printf("  power %d: %.6e\n",it,lam);
    }
    double c_shift=1.05*lam;
    /* Subspace iteration k=5 */
    int k1=5;
    dcomplex*Q=calloc(n_c1*k1,16),*W=calloc(n_c1*k1,16);
    srand(456);for(int i=0;i<n_c1*k1;i++)Q[i]=(rand()/(double)RAND_MAX-.5)+I*(rand()/(double)RAND_MAX-.5);
    ortho(Q,k1,n_c1);
    for(int it=0;it<30;it++){
        for(int j=0;j<k1;j++){
            memset(v,0,n_c2*16);
            for(int64_t e=0;e<d1->nnz;e++)v[d1->rows[e]]+=d1->vals[e]*Q[j*n_c1+d1->cols[e]];
            memset(W+j*n_c1,0,n_c1*16);
            for(int64_t e=0;e<d1->nnz;e++)W[j*n_c1+d1->cols[e]]+=conj(d1->vals[e])*v[d1->rows[e]];
        }
        for(int i=0;i<n_c1*k1;i++)W[i]=c_shift*Q[i]-W[i];
        ortho(W,k1,n_c1);dcomplex*tmp=Q;Q=W;W=tmp;
        if((it+1)%10==0||it==29){
            /* Ritz */
            for(int j=0;j<k1;j++){
                memset(v,0,n_c2*16);
                for(int64_t e=0;e<d1->nnz;e++)v[d1->rows[e]]+=d1->vals[e]*Q[j*n_c1+d1->cols[e]];
                memset(W+j*n_c1,0,n_c1*16);
                for(int64_t e=0;e<d1->nnz;e++)W[j*n_c1+d1->cols[e]]+=conj(d1->vals[e])*v[d1->rows[e]];
            }
            dcomplex*G=malloc(k1*k1*16);dcomplex al=1,be=0;int kk=n_c1;
            zgemm_("C","N",&k1,&k1,&kk,&al,Q,&n_c1,W,&n_c1,&be,G,&k1);
            double*ev=malloc(k1*8);int lw=4*k1*k1,lrw=3*k1*k1,liw=7*k1,info;
            dcomplex*w2=malloc(lw*16);double*rw=malloc(lrw*8);int*iw=malloc(liw*4);
            zheevd_("N","U",&k1,G,&k1,ev,w2,&lw,rw,&lrw,iw,&liw,&info);
            printf("  iter %d: Ritz=[",it+1);for(int i=0;i<k1;i++)printf("%.4e ",ev[i]);printf("]\n");
            free(w2);free(rw);free(iw);free(G);free(ev);
        }
    }
    /* Final Ritz */
    for(int j=0;j<k1;j++){
        memset(v,0,n_c2*16);
        for(int64_t e=0;e<d1->nnz;e++)v[d1->rows[e]]+=d1->vals[e]*Q[j*n_c1+d1->cols[e]];
        memset(W+j*n_c1,0,n_c1*16);
        for(int64_t e=0;e<d1->nnz;e++)W[j*n_c1+d1->cols[e]]+=conj(d1->vals[e])*v[d1->rows[e]];
    }
    dcomplex*G=malloc(k1*k1*16);dcomplex al=1,be=0;int kk=n_c1;
    zgemm_("C","N",&k1,&k1,&kk,&al,Q,&n_c1,W,&n_c1,&be,G,&k1);
    double*ev=malloc(k1*8);int lw=4*k1*k1,lrw=3*k1*k1,liw=7*k1,info;
    dcomplex*w2=malloc(lw*16);double*rw=malloc(lrw*8);int*iw=malloc(liw*4);
    zheevd_("N","U",&k1,G,&k1,ev,w2,&lw,rw,&lrw,iw,&liw,&info);
    free(w2);free(rw);free(iw);free(G);
    printf("\n  Final d1*d1 Ritz: ");for(int i=0;i<k1;i++)printf("%.4e ",ev[i]);printf("\n");
    double tol=ev[k1-1]*1e-8;int null_d1=0;
    for(int i=0;i<k1;i++)if(ev[i]<tol)null_d1++;
    printf("  nullity(d1)~%d rank(d1)~%d (expect 2, 19520)\n",null_d1,n_c1-null_d1);
    printf("  (%.1fs)\n",omp_get_wtime()-t);fflush(stdout);
    free(h);free(v);free(r);free(Q);free(W);free(ev);

    /* ---- Build d2 and verify d2∘d1=0 ---- */
    printf("\n[5] Building d2...\n");fflush(stdout);
    t=omp_get_wtime();
    /* For d2, we need C³ enumeration. C³ is huge (~138M rows).
       Instead of building the full d2 matrix, we verify d2∘d1=0
       by computing ||d2(d1(h))|| / ||d1(h)|| for random h.
       We apply d2 directly using the 13-term formula. */

    /* For d2, we need to know which C² entries are f-type and g-type.
       c2_type[idx] = 0 for f, 1 for g.
       c2_tuple[idx*3..idx*3+2] = (a,b,l) for f or (c,j,k) for g. */

    int n_tests=5;double max_ratio=0;
    srand(789);
    for(int test=0;test<n_tests;test++){
        /* Random h ∈ C¹ */
        dcomplex*hv=calloc(n_c1,16);
        for(int i=0;i<n_c1;i++)hv[i]=(rand()/(double)RAND_MAX-.5)+I*(rand()/(double)RAND_MAX-.5);

        /* v = d1 @ h (dense, n_c2) */
        dcomplex*vv=calloc(n_c2,16);
        for(int64_t e=0;e<d1->nnz;e++)vv[d1->rows[e]]+=d1->vals[e]*hv[d1->cols[e]];
        double nv2=0;for(int i=0;i<n_c2;i++)nv2+=creal(conj(vv[i])*vv[i]);

        /* w = d2 @ v: apply 13-term formula to each C² entry */
        /* C³ has 3 components: h-rows(a,b,c,t), m-rows(a,b,j,k), c-rows(c,j,k,l).
           We compute ||w||^2 directly without storing w.
           For each C² entry (flat idx i), we know its type and tuple:
             f-type (a,b,l): apply ∂^h f (4 terms → h-rows) + ∂^c f (3 terms → m-rows)
             g-type (c,j,k): apply ∂^h g (3 terms → m-rows) + ∂^c g (3 terms → c-rows)
           Each term contributes to ||w||^2 as |coeff * vv[i]|^2.
           BUT this OVERESTIMATES ||w||^2 because different entries can cancel.
           For a proper check, we'd need to accumulate w in a hash map.
           
           For the chain check, let's just compute ||d2(v)|| for the f-part only. */

        /* Actually, the proper way: compute w = d2 @ v as a sparse vector.
           For each C² entry i with value vv[i]:
             If f-type (af,bf,lf):
               ∂^h f: 4 terms producing h-row entries (a,b,c,t)
               ∂^c f: 3 terms producing m-row entries (a,b,j,k)
             If g-type (cf,jf,kf):
               ∂^h g: 3 terms producing m-row entries (a,b,j,k)
               ∂^c g: 3 terms producing c-row entries (c,j,k,l)
           
           We compute ||w||^2 by accumulating into a hash map for C³.
           But C³ has 138M entries — too large for a hash map.
           
           Alternative: compute ||d2(v)||^2 = Σ_{rows r} |w_r|^2.
           For each C² entry i, the terms contribute to specific C³ rows.
           Instead of forming w, compute ||w||^2 directly:
             ||w||^2 = Σ_i Σ_j <d2[:,i], d2[:,j]> * conj(vv[i]) * vv[j]
           This is vv* @ (d2* @ d2) @ vv = ||d2 @ vv||^2.
           But d2* @ d2 is n_c2 × n_c2 = 9.4M × 9.4M — too large.
           
           Simplest: just compute d2 @ vv as a sum of contributions,
           using a temporary hash map for C³.  The hash map will have
           at most (nnz per column of d2) * n_c2 entries, but each
           C² entry contributes ~7 terms, so ~7 * 7.9M = 55M entries.
           With a 128M hash, this fits.
           
           For now, let's just check the f-part (∂^h f + ∂^c f) separately.
           If the f-part gives ||w||/||v|| ≈ 0, the formula is likely correct. */

        /* Compute ||∂^h f(v)||^2 for the f-part only */
        /* For each f-type entry (af,bf,lf) with value vv[i]:
           T1: a·f(b,c) → h-row (a,af,bf,t) with coeff mult_bar[t,a,lf]*vv[i]
               → contributes |mult_bar[t,a,lf]*vv[i]|^2 to ||w||^2
           But this doesn't account for cross-term cancellation.
           We need to ACCUMULATE w, not just sum |coeff|^2. */

        /* Let me use a simpler approach: compute d2@v for just ONE
           f-type entry and check the formula manually. */
        if(test==0){
            /* Find the first f-type entry with nonzero v */
            int found=-1;
            for(int i=0;i<n_c2;i++){
                if(c2_type[i]==0&&cabs(vv[i])>1e-10){found=i;break;}
            }
            if(found>=0){
                int af=c2_tuple[found*3],bf=c2_tuple[found*3+1],lf=c2_tuple[found*3+2];
                printf("  Test f-entry: (%d,%d,%d) val=%.4e\n",af,bf,lf,vv[found]);
                /* T1: mult_bar[t,a,lf] for all (a,t) */
                double t1_norm=0;
                for(int i=0;i<n_mb_by_b[lf];i++){int idx=mb_by_b[lf][i];
                    t1_norm+=cabs(mb_val(idx)*vv[found]);}
                printf("  T1 (a·f(b,c)): %d terms, sum|coeff*val|=%.4e\n",n_mb_by_b[lf],t1_norm);
                /* T2: mult_bar[af,a,b] */
                double t2_norm=0;
                for(int i=0;i<n_mb_by_l[af];i++){int idx=mb_by_l[af][i];
                    t2_norm+=cabs(mb_val(idx)*vv[found]);}
                printf("  T2 (-f(ab,c)): %d terms, sum|coeff*val|=%.4e\n",n_mb_by_l[af],t2_norm);
                /* T3: mult_bar[bf,b,c] */
                double t3_norm=0;
                for(int i=0;i<n_mb_by_l[bf];i++){int idx=mb_by_l[bf][i];
                    t3_norm+=cabs(mb_val(idx)*vv[found]);}
                printf("  T3 (f(a,bc)): %d terms, sum|coeff*val|=%.4e\n",n_mb_by_l[bf],t3_norm);
                /* T4: mult_bar[t,lf,c] */
                double t4_norm=0;
                for(int i=0;i<n_mb_by_a[lf];i++){int idx=mb_by_a[lf][i];
                    t4_norm+=cabs(mb_val(idx)*vv[found]);}
                printf("  T4 (-f(a,b)·c): %d terms, sum|coeff*val|=%.4e\n",n_mb_by_a[lf],t4_norm);
            }
        }

        free(hv);free(vv);
        printf("  test %d: ||v||=%.4e (d2 norm computation needs C3 hash)\n",test,sqrt(nv2));
    }

    printf("\n  d1 nnz: %lld\n",(long long)d1->nnz);
    printf("  (d2∘d1 check requires C³ accumulation — TODO)\n");
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
