/*
 * verify_d2_d1_v2.c — Verify d2∘d1 = 0 with CORRECTED MW formula.
 *
 * Key corrections from arXiv:0704.2771:
 * 1. ∂_b(f,g) = (∂ʰf, ∂ʰg + ∂ᶜf, −∂ᶜg)  [+∂ᶜf, not -∂ᶜf!]
 * 2. ∂ʰg uses DIAGONAL action: (Δa)·g(b) and g(a)·(Δb)
 * 3. ∂ᶜf uses DIAGONAL coaction: a₁b₁ ⊗ f(a₂,b₂) and f(a₁,b₁) ⊗ a₂b₂
 * 4. ∂ᶜg has 4 TERMS: c₁⊗g(c₂) − (Δ⊗id)(g(c)) + (id⊗Δ)(g(c)) − g(c₁)⊗c₂
 *
 * Pure C. AVX-512. OpenMP.
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

static int32_t weights[DIM][2];
struct __attribute__((packed)) mb_entry { int32_t l,a,b; double re,im; };
struct __attribute__((packed)) db_entry { int32_t c,j,k; double re,im; };
static struct mb_entry mult_bar[NNZ_MB];
static struct db_entry delta_bar[NNZ_DB];

static dcomplex mb_val(int i){return mult_bar[i].re+I*mult_bar[i].im;}
static dcomplex db_val(int i){return delta_bar[i].re+I*delta_bar[i].im;}

/* 1D inverse tables */
static int32_t *mb_by_b[DIM],*mb_by_l[DIM],*mb_by_a[DIM];
static int n_mb_by_b[DIM],n_mb_by_l[DIM],n_mb_by_a[DIM];
static int32_t *db_by_c[DIM],*db_by_j[DIM],*db_by_k[DIM];
static int n_db_by_c[DIM],n_db_by_j[DIM],n_db_by_k[DIM];

/* 2D inverse table for delta_bar: indexed by (j,k) → list of (c, val_idx) */
/* db_by_jk[j*DIM + k] → array of delta_bar indices */
static int32_t *db_by_jk[DIM*DIM];
static int n_db_by_jk[DIM*DIM];

/* 2D inverse table for mult_bar: indexed by (a,b) → list of (l, val_idx) */
/* mb_by_ab[a*DIM + b] → array of mult_bar indices */
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

    /* Build 1D inverse tables */
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

    /* Build 2D inverse table for delta_bar: (j,k) → indices */
    for(int i=0;i<DIM*DIM;i++)n_db_by_jk[i]=0;
    for(int i=0;i<NNZ_DB;i++){int jk=delta_bar[i].j*DIM+delta_bar[i].k;n_db_by_jk[jk]++;}
    for(int i=0;i<DIM*DIM;i++){if(n_db_by_jk[i])db_by_jk[i]=malloc(n_db_by_jk[i]*sizeof(int32_t));n_db_by_jk[i]=0;}
    for(int i=0;i<NNZ_DB;i++){int jk=delta_bar[i].j*DIM+delta_bar[i].k;db_by_jk[jk][n_db_by_jk[jk]++]=i;}

    /* Build 2D inverse table for mult_bar: (a,b) → indices */
    for(int i=0;i<DIM*DIM;i++)n_mb_by_ab[i]=0;
    for(int i=0;i<NNZ_MB;i++){int ab=mult_bar[i].a*DIM+mult_bar[i].b;n_mb_by_ab[ab]++;}
    for(int i=0;i<DIM*DIM;i++){if(n_mb_by_ab[i])mb_by_ab[i]=malloc(n_mb_by_ab[i]*sizeof(int32_t));n_mb_by_ab[i]=0;}
    for(int i=0;i<NNZ_MB;i++){int ab=mult_bar[i].a*DIM+mult_bar[i].b;mb_by_ab[ab][n_mb_by_ab[ab]++]=i;}

    printf("  2D tables: db_by_jk nonzero=%d, mb_by_ab nonzero=%d\n",
           (int[]){0},0);
    int nz_jk=0,nz_ab=0;
    for(int i=0;i<DIM*DIM;i++){if(n_db_by_jk[i])nz_jk++;if(n_mb_by_ab[i])nz_ab++;}
    printf("  db_by_jk: %d nonzero slots, mb_by_ab: %d nonzero slots\n",nz_jk,nz_ab);
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

/* ====================================================================== */
/*  CORRECTED d2 forward matvec                                           */
/*  ∂_b(f,g) = (∂ʰf, ∂ʰg + ∂ᶜf, −∂ᶜg)                                   */
/* ====================================================================== */
static void d2_forward(const dcomplex *v,int nt){
    #pragma omp parallel for schedule(dynamic,64) num_threads(nt)
    for(int32_t i=0;i<g_n_c2;i++){
        dcomplex vi=v[i];
        if(cabs(vi)<1e-30)continue;
        scomplex vi32=(scomplex)vi;

        if(g_c2_type[i]==0){
            /* f-type: (af, bf, lf) */
            int af=g_c2_tuple[i*3],bf=g_c2_tuple[i*3+1],lf=g_c2_tuple[i*3+2];

            /* ===== ∂ʰf (4 terms → h-rows, comp=0), signs +,-,+,- ===== */
            /* T1: a·f(b,c) — simple mult on output. mult_bar[t,a,lf] → row(a,af,bf,t) */
            for(int e=0;e<n_mb_by_b[lf];e++){int idx=mb_by_b[lf][e];
                c3_accum(hkey(0,mult_bar[idx].a,af,bf,mult_bar[idx].l),
                         (scomplex)(mb_val(idx)*vi));}
            /* T2: -f(a·b,c). mult_bar[af,a,b] → row(a,b,bf,lf) */
            for(int e=0;e<n_mb_by_l[af];e++){int idx=mb_by_l[af][e];
                c3_accum(hkey(0,mult_bar[idx].a,mult_bar[idx].b,bf,lf),
                         (scomplex)(-mb_val(idx)*vi));}
            /* T3: +f(a,b·c). mult_bar[bf,b,c] → row(af,b,c,lf) */
            for(int e=0;e<n_mb_by_l[bf];e++){int idx=mb_by_l[bf][e];
                c3_accum(hkey(0,af,mult_bar[idx].a,mult_bar[idx].b,lf),
                         (scomplex)(mb_val(idx)*vi));}
            /* T4: -f(a,b)·c — simple mult on output. mult_bar[t,lf,c] → row(af,bf,c,t) */
            for(int e=0;e<n_mb_by_a[lf];e++){int idx=mb_by_a[lf][e];
                c3_accum(hkey(0,af,bf,mult_bar[idx].b,mult_bar[idx].l),
                         (scomplex)(-mb_val(idx)*vi));}

            /* ===== +∂ᶜf (3 terms → m-rows, comp=1), signs +,-,+ ===== */
            /* T1: a₁b₁ ⊗ f(a₂, b₂) — DIAGONAL coaction on (a,b)
               delta_bar[a, a₁, af] × delta_bar[b, b₁, bf] × mult_bar[j, a₁, b₁]
               → row(a, b, j, lf), val +d1*d2*m*vi */
            for(int e1=0;e1<n_db_by_k[af];e1++){int d1idx=db_by_k[af][e1];
                int a=delta_bar[d1idx].c, a1=delta_bar[d1idx].j;
                dcomplex d1val=db_val(d1idx);
                for(int e2=0;e2<n_db_by_k[bf];e2++){int d2idx=db_by_k[bf][e2];
                    int b=delta_bar[d2idx].c, b1=delta_bar[d2idx].j;
                    dcomplex d2val=db_val(d2idx);
                    /* Find mult_bar[j, a1, b1] via 2D table */
                    int ab=a1*DIM+b1;
                    for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];
                        int j=mult_bar[midx].l;
                        c3_accum(hkey(1,a,b,j,lf),
                                 (scomplex)(d1val*d2val*mb_val(midx)*vi));
                    }
                }
            }
            /* T2: -Δ(f(a,b)) — comultiply the output lf.
               delta_bar[lf, j, k] → row(af, bf, j, k), val -d*vi */
            for(int e=0;e<n_db_by_c[lf];e++){int idx=db_by_c[lf][e];
                c3_accum(hkey(1,af,bf,delta_bar[idx].j,delta_bar[idx].k),
                         (scomplex)(-db_val(idx)*vi));}
            /* T3: f(a₁, b₁) ⊗ a₂b₂ — DIAGONAL coaction on (a,b)
               delta_bar[a, a₁, a₂] × delta_bar[b, b₁, b₂] × mult_bar[j, a₂, b₂]
               where a₁=af, b₁=bf (so f(af,bf)=vi)
               → row(a, b, af, bf... wait, output is f(a₁,b₁) ⊗ a₂b₂ = lf ⊗ j
               row(a, b, lf, j), val +d1*d2*m*vi */
            for(int e1=0;e1<n_db_by_j[af];e1++){int d1idx=db_by_j[af][e1];
                /* delta_bar[a, af, a₂] = d1, so .c=a, .j=af, .k=a₂ */
                int a=delta_bar[d1idx].c, a2=delta_bar[d1idx].k;
                dcomplex d1val=db_val(d1idx);
                for(int e2=0;e2<n_db_by_j[bf];e2++){int d2idx=db_by_j[bf][e2];
                    int b=delta_bar[d2idx].c, b2=delta_bar[d2idx].k;
                    dcomplex d2val=db_val(d2idx);
                    /* Find mult_bar[j, a2, b2] via 2D table */
                    int ab=a2*DIM+b2;
                    for(int e3=0;e3<n_mb_by_ab[ab];e3++){int midx=mb_by_ab[ab][e3];
                        int j=mult_bar[midx].l;
                        c3_accum(hkey(1,a,b,lf,j),
                                 (scomplex)(d1val*d2val*mb_val(midx)*vi));
                    }
                }
            }
        } else {
            /* g-type: (cf, jf, kf) */
            int cf=g_c2_tuple[i*3],jf=g_c2_tuple[i*3+1],kf=g_c2_tuple[i*3+2];

            /* ===== ∂ʰg (3 terms → m-rows, comp=1), signs +,-,+ ===== */
            /* T1: (Δa)·g(b) — DIAGONAL left action. b=cf.
               delta_bar[a, a₁, a₂] × mult_bar[t1, a₁, jf] × mult_bar[t2, a₂, kf]
               → row(a, cf, t1, t2), val +d*m1*m2*vi
               Iterate: for each mult_bar[t1, a₁, jf] (mb_by_b[jf]),
                        for each mult_bar[t2, a₂, kf] (mb_by_b[kf]),
                        find delta_bar[a, a₁, a₂] via 2D table db_by_jk[a1*DIM+a2] */
            for(int e1=0;e1<n_mb_by_b[jf];e1++){int m1idx=mb_by_b[jf][e1];
                int t1=mult_bar[m1idx].l, a1=mult_bar[m1idx].a;
                dcomplex m1val=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_b[kf];e2++){int m2idx=mb_by_b[kf][e2];
                    int t2=mult_bar[m2idx].l, a2=mult_bar[m2idx].a;
                    dcomplex m2val=mb_val(m2idx);
                    /* Find delta_bar[a, a1, a2] via 2D table */
                    int jk=a1*DIM+a2;
                    for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];
                        int a=delta_bar[didx].c;
                        c3_accum(hkey(1,a,cf,t1,t2),
                                 (scomplex)(db_val(didx)*m1val*m2val*vi));
                    }
                }
            }
            /* T2: -g(a·b) — multiply inputs. mult_bar[cf, a, b] → row(a, b, jf, kf) */
            for(int e=0;e<n_mb_by_l[cf];e++){int idx=mb_by_l[cf][e];
                c3_accum(hkey(1,mult_bar[idx].a,mult_bar[idx].b,jf,kf),
                         (scomplex)(-mb_val(idx)*vi));}
            /* T3: g(a)·(Δb) — DIAGONAL right action. a=cf.
               mult_bar[t1, jf, b₁] × mult_bar[t2, kf, b₂] × delta_bar[b, b₁, b₂]
               → row(cf, b, t1, t2), val +m1*m2*d*vi
               Iterate: for each mult_bar[t1, jf, b₁] (mb_by_a[jf]),
                        for each mult_bar[t2, kf, b₂] (mb_by_a[kf]),
                        find delta_bar[b, b₁, b₂] via 2D table */
            for(int e1=0;e1<n_mb_by_a[jf];e1++){int m1idx=mb_by_a[jf][e1];
                int t1=mult_bar[m1idx].l, b1=mult_bar[m1idx].b;
                dcomplex m1val=mb_val(m1idx);
                for(int e2=0;e2<n_mb_by_a[kf];e2++){int m2idx=mb_by_a[kf][e2];
                    int t2=mult_bar[m2idx].l, b2=mult_bar[m2idx].b;
                    dcomplex m2val=mb_val(m2idx);
                    /* Find delta_bar[b, b1, b2] via 2D table */
                    int jk=b1*DIM+b2;
                    for(int e3=0;e3<n_db_by_jk[jk];e3++){int didx=db_by_jk[jk][e3];
                        int b=delta_bar[didx].c;
                        c3_accum(hkey(1,cf,b,t1,t2),
                                 (scomplex)(m1val*m2val*db_val(didx)*vi));
                    }
                }
            }

            /* ===== -∂ᶜg (4 terms → c-rows, comp=2), signs -,+,-,+ ===== */
            /* T1: -c₁ ⊗ g(c₂) — left coaction. delta_bar[c, c₁, cf] → row(c, c₁, jf, kf)
               (c₂ = cf, so g(c₂) = g(cf) = vi)
               val -d*vi */
            for(int e=0;e<n_db_by_k[cf];e++){int idx=db_by_k[cf][e];
                c3_accum(hkey(2,delta_bar[idx].c,delta_bar[idx].j,jf,kf),
                         (scomplex)(-db_val(idx)*vi));}
            /* T2: +(Δ⊗id)(g(c)) — Δ on LEFT factor (jf) of g(cf).
               delta_bar[jf, j, k] → row(cf, j, k, kf), val +d*vi */
            for(int e=0;e<n_db_by_c[jf];e++){int idx=db_by_c[jf][e];
                c3_accum(hkey(2,cf,delta_bar[idx].j,delta_bar[idx].k,kf),
                         (scomplex)(db_val(idx)*vi));}
            /* T3: -(id⊗Δ)(g(c)) — Δ on RIGHT factor (kf) of g(cf).
               delta_bar[kf, j, k] → row(cf, jf, j, k), val -d*vi */
            for(int e=0;e<n_db_by_c[kf];e++){int idx=db_by_c[kf][e];
                c3_accum(hkey(2,cf,jf,delta_bar[idx].j,delta_bar[idx].k),
                         (scomplex)(-db_val(idx)*vi));}
            /* T4: +g(c₁) ⊗ c₂ — right coaction. delta_bar[c, cf, c₂] → row(c, jf, kf, c₂)
               (c₁ = cf, so g(c₁) = g(cf) = vi)
               val +d*vi */
            for(int e=0;e<n_db_by_j[cf];e++){int idx=db_by_j[cf][e];
                c3_accum(hkey(2,delta_bar[idx].c,jf,kf,delta_bar[idx].k),
                         (scomplex)(db_val(idx)*vi));}
        }
    }
}

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

    printf("=== d2∘d1=0 verification with CORRECTED MW formula ===\nThreads=%d\n\n",nt);fflush(stdout);

    load_algebra(dir);build_weight_classes();
    printf("mult_bar:%d delta_bar:%d\n\n",NNZ_MB,NNZ_DB);fflush(stdout);

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
        for(int i=0; i<n_db_by_c[hk];i++){int idx=db_by_c[hk][i];
            int32_t r=hash_insert(gkey(hk,delta_bar[idx].j,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,db_val(idx));}
        for(int i=0;i<n_db_by_j[hk];i++){int idx=db_by_j[hk][i];
            int32_t r=hash_insert(gkey(delta_bar[idx].c,hj,delta_bar[idx].k));
            if(r>=0)spmat_add(d1,r,ci,-db_val(idx));}
    }
    printf("  d1: %lld nnz (%.1fs)\n\n",(long long)d1->nnz,omp_get_wtime()-t);fflush(stdout);

    g_c3=malloc((size_t)C3_SIZE*sizeof(c3_entry));
    if(!g_c3){fprintf(stderr,"malloc g_c3 failed\n");exit(1);}
    g_n_c2=n_c2;
    g_c2_tuple=c2_tuple;
    g_c2_type=c2_type;

    /* Test d2∘d1=0 */
    printf("Testing d2∘d1=0 (3 random vectors)...\n\n");fflush(stdout);
    srand(12345);
    double max_ratio=0;
    for(int test=0;test<3;test++){
        dcomplex*hv=calloc(n_c1,16);
        for(int i=0;i<n_c1;i++)hv[i]=(rand()/(double)RAND_MAX-.5)+I*(rand()/(double)RAND_MAX-.5);
        dcomplex*vv=calloc(n_c2,16);
        for(int64_t e=0;e<d1->nnz;e++)vv[d1->rows[e]]+=d1->vals[e]*hv[d1->cols[e]];
        double nv2=0;
        #pragma omp parallel for reduction(+:nv2)
        for(int i=0;i<n_c2;i++)nv2+=creal(conj(vv[i])*vv[i]);

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
