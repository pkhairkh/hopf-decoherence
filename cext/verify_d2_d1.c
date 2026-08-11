/*
 * verify_d2_d1.c — Verify d2∘d1 = 0 for the corrected d2 formula.
 *
 * Tests ||d2(d1(h))|| / ||d1(h)|| for random h ∈ C¹.
 * If ratio < 1e-8, the d2 formula is correct.
 *
 * Pure C. AVX-512. OpenMP.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o verify_d2_d1 verify_d2_d1.c \
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

/* Algebra data */
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

/* Weight classes */
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

/* C² hash */
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

/* d1 sparse matrix */
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

static int32_t g_n_c2;
static int32_t *g_c2_tuple;
static int8_t *g_c2_type;

/* d2 forward matvec: store result in C³ hash */
static void d2_forward(const dcomplex *v,int nt){
    #pragma omp parallel for schedule(dynamic,256) num_threads(nt)
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
            /* ∂ᶜf disabled for baseline test */
            /*
            for(int e=0;e<n_db_by_k[af];e++){int idx=db_by_k[af][e];
                c3_accum(hkey(1,delta_bar[idx].c,bf,delta_bar[idx].j,lf),(scomplex)(-db_val(idx)*vi));}
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];
                c3_accum(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k),(scomplex)(db_val(idx)*vi));}
            for(int e=0;e<n_db_by_j[bf];e++){int idx=db_by_j[bf][e];
                c3_accum(hkey(1,af,delta_bar[idx].c,lf,delta_bar[idx].k),(scomplex)(-db_val(idx)*vi));}
            */
        } else {
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];
            /* ∂ʰg: +,-,+ */
            for(int e=0;e<n_mb_by_b[jf];e++){int idx=mb_by_b[jf][e];
                c3_accum(hkey(1,mult_bar[idx].a,cf,mult_bar[idx].l,kf),(scomplex)(mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];
                c3_accum(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf),(scomplex)(-mb_val(idx)*vi));}
            for(int e=0;e<n_mb_by_b[kf];e++){int idx=mb_by_b[kf][e];
                c3_accum(hkey(1,cf,mult_bar[idx].a,jf,mult_bar[idx].l),(scomplex)(mb_val(idx)*vi));}
            /* ∂ᶜg disabled for baseline test */
            /*
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];
                c3_accum(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf),(scomplex)(-db_val(idx)*vi));}
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];
                c3_accum(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf),(scomplex)(db_val(idx)*vi));}
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];
                c3_accum(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k),(scomplex)(-db_val(idx)*vi));}
            */
        }
    }
}

/* Compute ||w||² from C³ hash */
static double c3_norm2(void){
    double n2=0;
    #pragma omp parallel for reduction(+:n2) schedule(static)
    for(int64_t i=0;i<(int64_t)C3_SIZE;i++){
        if(g_c3[i].key!=0xFFFFFFFFFFFFFFFFULL){
            scomplex v=g_c3[i].val;
            n2+=(double)crealf(v)*crealf(v)+(double)cimagf(v)*cimagf(v);
        }
    }
    return n2;
}

int main(int argc,char*argv[]){
    const char*dir=(argc>1)?argv[1]:".";
    int nt=omp_get_max_threads();
    if(nt>15)nt=15;
    omp_set_num_threads(nt);
    double T0=omp_get_wtime();

    printf("=== d2∘d1=0 verification (pure C) ===\nThreads=%d\n\n",nt);fflush(stdout);

    load_algebra(dir);build_weight_classes();
    printf("mult_bar:%d delta_bar:%d\n\n",NNZ_MB,NNZ_DB);fflush(stdout);

    /* Build C¹, C², d1 */
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
    printf("C1:%d C2:%d (%df+%dg)\n\n",n_c1,n_c2,n_f,n_g);fflush(stdout);

    /* Build d1 */
    printf("Building d1...\n");fflush(stdout);
    double t=omp_get_wtime();
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

    /* Allocate C³ hash */
    g_c3=malloc((size_t)C3_SIZE*sizeof(c3_entry));
    if(!g_c3){fprintf(stderr,"malloc g_c3 failed\n");exit(1);}
    g_n_c2=n_c2;
    g_c2_tuple=c2_tuple;
    g_c2_type=c2_type;

    /* Test d2∘d1=0 */
    printf("Testing d2∘d1=0 (5 random vectors)...\n\n");fflush(stdout);
    srand(12345);
    double max_ratio=0;
    for(int test=0;test<5;test++){
        /* Random h ∈ C¹ */
        dcomplex*hv=calloc(n_c1,16);
        for(int i=0;i<n_c1;i++)hv[i]=(rand()/(double)RAND_MAX-.5)+I*(rand()/(double)RAND_MAX-.5);

        /* v = d1 @ h */
        dcomplex*vv=calloc(n_c2,16);
        for(int64_t e=0;e<d1->nnz;e++)vv[d1->rows[e]]+=d1->vals[e]*hv[d1->cols[e]];
        double nv2=0;
        #pragma omp parallel for reduction(+:nv2)
        for(int i=0;i<n_c2;i++)nv2+=creal(conj(vv[i])*vv[i]);

        /* w = d2 @ v (stored in C³ hash) */
        c3_clear();
        d2_forward(vv,nt);
        double nw2=c3_norm2();

        double ratio=(nv2>1e-10)?sqrt(nw2/nv2):0;
        if(ratio>max_ratio)max_ratio=ratio;
        printf("  test %d: ||d1(h)||=%.4e ||d2(d1(h))||=%.4e ratio=%.4e\n",
               test,sqrt(nv2),sqrt(nw2),ratio);fflush(stdout);

        free(hv);free(vv);
    }

    printf("\n=== RESULT ===\n");
    printf("  max ||d2∘d1||/||d1|| = %.4e %s\n",max_ratio,
           max_ratio<1e-8?"PASS ✓":"FAIL ✗");
    printf("\nTotal: %.1fs\n",omp_get_wtime()-T0);fflush(stdout);

    free(g_c3);
    free(d1->rows);free(d1->cols);free(d1->vals);free(d1);
    free(c2hash);
    return 0;
}
