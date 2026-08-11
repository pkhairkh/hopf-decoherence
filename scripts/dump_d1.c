/*
 * dump_d1.c — Dump first N d1 entries for comparison with Python export.
 * Build: gcc -O3 -fopenmp -o dump_d1 dump_d1.c -lm
 * Run:   ./dump_d1 [data_dir] [n_entries]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef double _Complex dcomplex;
#define ELL 3
#define DIM 243
#define NNZ_MB 27671
#define NNZ_DB 2647

static int32_t weights[DIM][2];
struct __attribute__((packed)) mb_entry { int32_t l,a,b; double re,im; };
struct __attribute__((packed)) db_entry { int32_t c,j,k; double re,im; };
static struct mb_entry mult_bar[NNZ_MB];
static struct db_entry delta_bar[NNZ_DB];
static int32_t *mb_by_b[DIM],*mb_by_l[DIM],*mb_by_a[DIM];
static int n_mb_by_b[DIM],n_mb_by_l[DIM],n_mb_by_a[DIM];
static int32_t *db_by_c[DIM],*db_by_j[DIM],*db_by_k[DIM];
static int n_db_by_c[DIM],n_db_by_j[DIM],n_db_by_k[DIM];
static dcomplex mb_val(int i){return mult_bar[i].re+I*mult_bar[i].im;}
static dcomplex db_val(int i){return delta_bar[i].re+I*delta_bar[i].im;}

static int32_t w2i_elems[ELL*ELL][81];
static int w2i_n[ELL*ELL];
static int wt_key(int w0,int w1){return w0*ELL+w1;}

#define HASH_SIZE (1<<26)
#define C2_MAX 20000000
typedef struct{uint64_t key;int32_t idx;}hentry;
static hentry *c2hash;static int32_t c2count;
static int32_t *c2_tuple;static int8_t *c2_type;
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
    if(c2count>=C2_MAX){fprintf(stderr,"FATAL: c2_count overflow\n");exit(1);}
    c2hash[h].key=key;c2hash[h].idx=c2count++;return c2hash[h].idx;
}
static int32_t hash_lookup(uint64_t key){
    uint64_t h=key%HASH_SIZE;
    while(c2hash[h].idx!=-1){if(c2hash[h].key==key)return c2hash[h].idx;h=(h+1)%HASH_SIZE;}
    return -1;
}

int main(int argc,char*argv[]){
    const char*dir=(argc>1)?argv[1]:".";
    int dump_n=(argc>2)?atoi(argv[2]):100;
    char p[512];FILE*f;

    snprintf(p,sizeof(p),"%s/weights.bin",dir);f=fopen(p,"rb");fread(weights,sizeof(int32_t),DIM*2,f);fclose(f);
    snprintf(p,sizeof(p),"%s/mult_bar.bin",dir);f=fopen(p,"rb");fread(mult_bar,sizeof(struct mb_entry),NNZ_MB,f);fclose(f);
    snprintf(p,sizeof(p),"%s/delta_bar.bin",dir);f=fopen(p,"rb");fread(delta_bar,sizeof(struct db_entry),NNZ_DB,f);fclose(f);

    for(int i=0;i<DIM;i++){n_mb_by_b[i]=n_mb_by_l[i]=n_mb_by_a[i]=0;n_db_by_c[i]=n_db_by_j[i]=n_db_by_k[i]=0;}
    for(int i=0;i<NNZ_MB;i++){n_mb_by_b[mult_bar[i].b]++;n_mb_by_l[mult_bar[i].l]++;n_mb_by_a[mult_bar[i].a]++;}
    for(int i=0;i<NNZ_DB;i++){n_db_by_c[delta_bar[i].c]++;n_db_by_j[delta_bar[i].j]++;n_db_by_k[delta_bar[i].k]++;}
    for(int i=0;i<DIM;i++){
        if(n_mb_by_b[i])mb_by_b[i]=malloc(n_mb_by_b[i]*4);
        if(n_mb_by_l[i])mb_by_l[i]=malloc(n_mb_by_l[i]*4);
        if(n_mb_by_a[i])mb_by_a[i]=malloc(n_mb_by_a[i]*4);
        if(n_db_by_c[i])db_by_c[i]=malloc(n_db_by_c[i]*4);
        if(n_db_by_j[i])db_by_j[i]=malloc(n_db_by_j[i]*4);
        if(n_db_by_k[i])db_by_k[i]=malloc(n_db_by_k[i]*4);
        n_mb_by_b[i]=n_mb_by_l[i]=n_mb_by_a[i]=0;
        n_db_by_c[i]=n_db_by_j[i]=n_db_by_k[i]=0;
    }
    for(int i=0;i<NNZ_MB;i++){mb_by_b[mult_bar[i].b][n_mb_by_b[mult_bar[i].b]++]=i;mb_by_l[mult_bar[i].l][n_mb_by_l[mult_bar[i].l]++]=i;mb_by_a[mult_bar[i].a][n_mb_by_a[mult_bar[i].a]++]=i;}
    for(int i=0;i<NNZ_DB;i++){db_by_c[delta_bar[i].c][n_db_by_c[delta_bar[i].c]++]=i;db_by_j[delta_bar[i].j][n_db_by_j[delta_bar[i].j]++]=i;db_by_k[delta_bar[i].k][n_db_by_k[delta_bar[i].k]++]=i;}

    for(int i=1;i<DIM;i++){int k=wt_key(weights[i][0],weights[i][1]);w2i_elems[k][w2i_n[k]++]=i;}

    /* Build C1 and C2 for shift (0,0) */
    int s0=0,s1=0;
    int32_t n_c1=0;int32_t c1j[20000],c1k[20000];
    for(int wj=0;wj<ELL*ELL;wj++)for(int wk=0;wk<ELL*ELL;wk++)
        if((wj/ELL-wk/ELL)%ELL==s0&&(wj%ELL-wk%ELL)%ELL==s1)
            for(int ji=0;ji<w2i_n[wj];ji++)for(int ki=0;ki<w2i_n[wk];ki++)
                {c1j[n_c1]=w2i_elems[wj][ji];c1k[n_c1]=w2i_elems[wk][ki];n_c1++;}

    hash_init();
    int32_t n_f=0,n_g=0;
    for(int wa=0;wa<ELL*ELL;wa++){if(!w2i_n[wa])continue;
        for(int wb=0;wb<ELL*ELL;wb++){if(!w2i_n[wb])continue;
            int wl=wt_key((s0+wa/ELL+wb/ELL)%ELL,(s1+wa%ELL+wb%ELL)%ELL);
            if(!w2i_n[wl])continue;
            for(int ai=0;ai<w2i_n[wa];ai++)for(int bi=0;bi<w2i_n[wb];bi++)for(int li=0;li<w2i_n[wl];li++){
                int a=w2i_elems[wa][ai],b=w2i_elems[wb][bi],l=w2i_elems[wl][li];
                int32_t idx=hash_insert(fkey(a,b,l));
                c2_tuple[idx*3]=a;c2_tuple[idx*3+1]=b;c2_tuple[idx*3+2]=l;c2_type[idx]=0;n_f++;
            }
        }
    }
    for(int wc=0;wc<ELL*ELL;wc++){if(!w2i_n[wc])continue;
        for(int wj=0;wj<ELL*ELL;wj++){if(!w2i_n[wj])continue;
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
    fprintf(stderr,"C1=%d C2=%d (%df+%dg)\n",n_c1,n_c2,n_f,n_g);

    /* Build d1 entries and dump them sorted by (row, col) */
    /* Collect all entries, sort, dump */
    typedef struct{int32_t row,col;double re,im;}entry;
    entry *entries=malloc(10000000*sizeof(entry));
    int n_entries=0;

    for(int ci=0;ci<n_c1;ci++){
        int hj=c1j[ci],hk=c1k[ci];
        /* T1 */
        for(int i=0;i<n_mb_by_b[hj];i++){int idx=mb_by_b[hj][i];
            int32_t r=hash_lookup(fkey(mult_bar[idx].a,hk,mult_bar[idx].l));
            if(r>=0){dcomplex v=mb_val(idx);entries[n_entries].row=r;entries[n_entries].col=ci;entries[n_entries].re=creal(v);entries[n_entries].im=cimag(v);n_entries++;}
        }
        /* T2 */
        for(int i=0;i<n_mb_by_l[hk];i++){int idx=mb_by_l[hk][i];
            int32_t r=hash_lookup(fkey(mult_bar[idx].a,mult_bar[idx].b,hj));
            if(r>=0){dcomplex v=-mb_val(idx);entries[n_entries].row=r;entries[n_entries].col=ci;entries[n_entries].re=creal(v);entries[n_entries].im=cimag(v);n_entries++;}
        }
        /* T3 */
        for(int i=0;i<n_mb_by_a[hj];i++){int idx=mb_by_a[hj][i];
            int32_t r=hash_lookup(fkey(hk,mult_bar[idx].b,mult_bar[idx].l));
            if(r>=0){dcomplex v=mb_val(idx);entries[n_entries].row=r;entries[n_entries].col=ci;entries[n_entries].re=creal(v);entries[n_entries].im=cimag(v);n_entries++;}
        }
        /* T4 (∂^c T1) */
        for(int i=0;i<n_db_by_k[hk];i++){int idx=db_by_k[hk][i];
            int32_t r=hash_lookup(gkey(delta_bar[idx].c,delta_bar[idx].j,hj));
            if(r>=0){dcomplex v=db_val(idx);entries[n_entries].row=r;entries[n_entries].col=ci;entries[n_entries].re=creal(v);entries[n_entries].im=cimag(v);n_entries++;}
        }
        /* T5 (∂^c T2) */
        for(int i=0;i<n_db_by_c[hj];i++){int idx=db_by_c[hj][i];
            int32_t r=hash_lookup(gkey(hk,delta_bar[idx].j,delta_bar[idx].k));
            if(r>=0){dcomplex v=-db_val(idx);entries[n_entries].row=r;entries[n_entries].col=ci;entries[n_entries].re=creal(v);entries[n_entries].im=cimag(v);n_entries++;}
        }
        /* T6 (∂^c T3) */
        for(int i=0;i<n_db_by_j[hk];i++){int idx=db_by_j[hk][i];
            int32_t r=hash_lookup(gkey(delta_bar[idx].c,hj,delta_bar[idx].k));
            if(r>=0){dcomplex v=db_val(idx);entries[n_entries].row=r;entries[n_entries].col=ci;entries[n_entries].re=creal(v);entries[n_entries].im=cimag(v);n_entries++;}
        }
    }

    fprintf(stderr,"d1: %d entries\n",n_entries);

    /* Sort by (row, col) */
    /* Simple: dump unsorted, let Python sort */
    printf("row,col,re,im\n");
    int limit = (n_entries < dump_n) ? n_entries : dump_n;
    for(int i=0;i<limit;i++){
        printf("%d,%d,%.15e,%.15e\n",entries[i].row,entries[i].col,entries[i].re,entries[i].im);
    }
    fprintf(stderr,"Dumped %d of %d entries\n",limit,n_entries);

    return 0;
}
