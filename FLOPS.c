#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#define N_FA_ARRAYS 8 // from fa0 to fa7
#define MAXFLOPS_ITERS 1000000000
#define LOOP_COUNT N_THREADS*N_FA_ARRAYS
#define FLOPS_ARRAY_SIZE LOOP_COUNT*N_THREADS*N_FA_ARRAYS

#ifdef AVX_512
  #define BYTES_IN_VECT 64
  #define TYPE __m512
  #define INTR1 _mm512_fmadd_ps
  #define INTR2 _mm512_set1_ps
  #define INTR3 _mm512_set_ps
  #define INTR4 _mm512_add_ps
  float sum(__m512 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<16;i++)res += val[i];
    return res;
  }
#elif defined AVX_256
  #define BYTES_IN_VECT 32
  #define TYPE __m256
  #define INTR1 _mm256_fmadd_ps
  #define INTR2 _mm256_set1_ps
  #define INTR3 _mm256_set_ps
  #define INTR4 _mm256_add_ps
  float sum(__m256 x) {
    float *val = (float*) &x;
    float res = 0.0;
    for(int i=0;i<8;i++)res += val[i];
    return res;
  }
#endif

struct fakeReg {
  TYPE reg;
  int trash[(3*64)/4];
};

//float fa[FLOPS_ARRAY_SIZE] __attribute__((aligned(64)));
//float fb[FLOPS_ARRAY_SIZE] __attribute__((aligned(64)));

float fa[FLOPS_ARRAY_SIZE];
float fb[FLOPS_ARRAY_SIZE];

struct fakeReg fa0[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fa1[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fa2[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fa3[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fa4[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fa5[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fa6[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fa7[N_THREADS] __attribute__((aligned(64)));

struct fakeReg fb0[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fb1[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fb2[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fb3[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fb4[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fb5[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fb6[N_THREADS] __attribute__((aligned(64)));
struct fakeReg fb7[N_THREADS] __attribute__((aligned(64)));

void compute(int index, TYPE mult) {
  for(long i=0; i<MAXFLOPS_ITERS; i++) {
      fa0[index].reg = INTR1(mult, fa0[index].reg, fb0[index].reg);
      fa1[index].reg = INTR1(mult, fa1[index].reg, fb1[index].reg);
      fa2[index].reg = INTR1(mult, fa2[index].reg, fb2[index].reg);
      fa3[index].reg = INTR1(mult, fa3[index].reg, fb3[index].reg);
      fa4[index].reg = INTR1(mult, fa4[index].reg, fb4[index].reg);
      fa5[index].reg = INTR1(mult, fa5[index].reg, fb5[index].reg);
      fa6[index].reg = INTR1(mult, fa6[index].reg, fb6[index].reg);
      fa7[index].reg = INTR1(mult, fa7[index].reg, fb7[index].reg);
  }
}

int main() {
  omp_set_num_threads(N_THREADS);
  int i = 0;
  int t = 0;
  TYPE mult = INTR2(0.1f);

  for(i=0; i<FLOPS_ARRAY_SIZE; i++) {
    fa[i] = (float)i + 0.1f;
    fb[i] = (float)i + 0.2f;
  }
  printf("%d\n",sizeof(fb7));

#ifdef AVX_256
  for(i=0;i<N_THREADS;i++) {
    fa0[i].reg = INTR3 (fa[LOOP_COUNT*i],fa[LOOP_COUNT*i+1],fa[LOOP_COUNT*i+2],fa[LOOP_COUNT*i+3],fa[LOOP_COUNT*i+4],fa[LOOP_COUNT*i+5],fa[LOOP_COUNT*i+6],fa[LOOP_COUNT*i+7]);
    fa1[i].reg = INTR3 (fa[LOOP_COUNT*i+8],fa[LOOP_COUNT*i+9],fa[LOOP_COUNT*i+10],fa[LOOP_COUNT*i+11],fa[LOOP_COUNT*i+12],fa[LOOP_COUNT*i+13],fa[LOOP_COUNT*i+14],fa[LOOP_COUNT*i+15]);
    fa2[i].reg = INTR3 (fa[LOOP_COUNT*i+16],fa[LOOP_COUNT*i+17],fa[LOOP_COUNT*i+18],fa[LOOP_COUNT*i+19],fa[LOOP_COUNT*i+20],fa[LOOP_COUNT*i+21],fa[LOOP_COUNT*i+22],fa[LOOP_COUNT*i+23]);
    fa3[i].reg = INTR3 (fa[LOOP_COUNT*i+24],fa[LOOP_COUNT*i+25],fa[LOOP_COUNT*i+26],fa[LOOP_COUNT*i+27],fa[LOOP_COUNT*i+28],fa[LOOP_COUNT*i+29],fa[LOOP_COUNT*i+30],fa[LOOP_COUNT*i+31]);
    fa4[i].reg = INTR3 (fa[LOOP_COUNT*i+32],fa[LOOP_COUNT*i+33],fa[LOOP_COUNT*i+34],fa[LOOP_COUNT*i+35],fa[LOOP_COUNT*i+36],fa[LOOP_COUNT*i+37],fa[LOOP_COUNT*i+38],fa[LOOP_COUNT*i+39]);
    fa5[i].reg = INTR3 (fa[LOOP_COUNT*i+40],fa[LOOP_COUNT*i+41],fa[LOOP_COUNT*i+42],fa[LOOP_COUNT*i+43],fa[LOOP_COUNT*i+44],fa[LOOP_COUNT*i+45],fa[LOOP_COUNT*i+46],fa[LOOP_COUNT*i+47]);
    fa6[i].reg = INTR3 (fa[LOOP_COUNT*i+48],fa[LOOP_COUNT*i+49],fa[LOOP_COUNT*i+50],fa[LOOP_COUNT*i+51],fa[LOOP_COUNT*i+52],fa[LOOP_COUNT*i+53],fa[LOOP_COUNT*i+54],fa[LOOP_COUNT*i+55]);
    fa7[i].reg = INTR3 (fa[LOOP_COUNT*i+56],fa[LOOP_COUNT*i+57],fa[LOOP_COUNT*i+58],fa[LOOP_COUNT*i+59],fa[LOOP_COUNT*i+60],fa[LOOP_COUNT*i+61],fa[LOOP_COUNT*i+62],fa[LOOP_COUNT*i+63]);

    fb0[i].reg = INTR3 (fb[LOOP_COUNT*i+0],fb[LOOP_COUNT*i+1],fb[LOOP_COUNT*i+2],fb[LOOP_COUNT*i+3],fb[LOOP_COUNT*i+4],fb[LOOP_COUNT*i+5],fb[LOOP_COUNT*i+6],fb[LOOP_COUNT*i+7]);
    fb1[i].reg = INTR3 (fb[LOOP_COUNT*i+8],fb[LOOP_COUNT*i+9],fb[LOOP_COUNT*i+10],fb[LOOP_COUNT*i+11],fb[LOOP_COUNT*i+12],fb[LOOP_COUNT*i+13],fb[LOOP_COUNT*i+14],fb[LOOP_COUNT*i+15]);
    fb2[i].reg = INTR3 (fb[LOOP_COUNT*i+16],fb[LOOP_COUNT*i+17],fb[LOOP_COUNT*i+18],fb[LOOP_COUNT*i+19],fb[LOOP_COUNT*i+20],fb[LOOP_COUNT*i+21],fb[LOOP_COUNT*i+22],fb[LOOP_COUNT*i+23]);
    fb3[i].reg = INTR3 (fb[LOOP_COUNT*i+24],fb[LOOP_COUNT*i+25],fb[LOOP_COUNT*i+26],fb[LOOP_COUNT*i+27],fb[LOOP_COUNT*i+28],fb[LOOP_COUNT*i+29],fb[LOOP_COUNT*i+30],fb[LOOP_COUNT*i+31]);
    fb4[i].reg = INTR3 (fb[LOOP_COUNT*i+32],fb[LOOP_COUNT*i+33],fb[LOOP_COUNT*i+34],fb[LOOP_COUNT*i+35],fb[LOOP_COUNT*i+36],fb[LOOP_COUNT*i+37],fb[LOOP_COUNT*i+38],fb[LOOP_COUNT*i+39]);
    fb5[i].reg = INTR3 (fb[LOOP_COUNT*i+40],fb[LOOP_COUNT*i+41],fb[LOOP_COUNT*i+42],fb[LOOP_COUNT*i+43],fb[LOOP_COUNT*i+44],fb[LOOP_COUNT*i+45],fb[LOOP_COUNT*i+46],fb[LOOP_COUNT*i+47]);
    fb6[i].reg = INTR3 (fb[LOOP_COUNT*i+48],fb[LOOP_COUNT*i+49],fb[LOOP_COUNT*i+50],fb[LOOP_COUNT*i+51],fb[LOOP_COUNT*i+52],fb[LOOP_COUNT*i+53],fb[LOOP_COUNT*i+54],fb[LOOP_COUNT*i+55]);
    fb7[i].reg = INTR3 (fb[LOOP_COUNT*i+56],fb[LOOP_COUNT*i+57],fb[LOOP_COUNT*i+58],fb[LOOP_COUNT*i+59],fb[LOOP_COUNT*i+60],fb[LOOP_COUNT*i+61],fb[LOOP_COUNT*i+62],fb[LOOP_COUNT*i+63]);
  }
#elif defined AVX_512
  for(i=0;i<N_THREADS;i++) {
    fa0[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i],fa[LOOP_COUNT*i+1],fa[LOOP_COUNT*i+2],fa[LOOP_COUNT*i+3],fa[LOOP_COUNT*i+4],fa[LOOP_COUNT*i+5],fa[LOOP_COUNT*i+6],fa[LOOP_COUNT*i+7],
                            fa[LOOP_COUNT*i+8],fa[LOOP_COUNT*i+9],fa[LOOP_COUNT*i+10],fa[LOOP_COUNT*i+11],fa[LOOP_COUNT*i+12],fa[LOOP_COUNT*i+13],fa[LOOP_COUNT*i+14],fa[LOOP_COUNT*i+15]);

    fa1[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i+16],fa[LOOP_COUNT*i+17],fa[LOOP_COUNT*i+18],fa[LOOP_COUNT*i+19],fa[LOOP_COUNT*i+20],fa[LOOP_COUNT*i+21],fa[LOOP_COUNT*i+22],fa[LOOP_COUNT*i+23],
                            fa[LOOP_COUNT*i+24],fa[LOOP_COUNT*i+25],fa[LOOP_COUNT*i+26],fa[LOOP_COUNT*i+27],fa[LOOP_COUNT*i+28],fa[LOOP_COUNT*i+29],fa[LOOP_COUNT*i+30],fa[LOOP_COUNT*i+31]);

    fa2[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i+32],fa[LOOP_COUNT*i+33],fa[LOOP_COUNT*i+34],fa[LOOP_COUNT*i+35],fa[LOOP_COUNT*i+36],fa[LOOP_COUNT*i+37],fa[LOOP_COUNT*i+38],fa[LOOP_COUNT*i+39],
                            fa[LOOP_COUNT*i+40],fa[LOOP_COUNT*i+41],fa[LOOP_COUNT*i+42],fa[LOOP_COUNT*i+43],fa[LOOP_COUNT*i+44],fa[LOOP_COUNT*i+45],fa[LOOP_COUNT*i+46],fa[LOOP_COUNT*i+47]);

    fa3[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i+48],fa[LOOP_COUNT*i+49],fa[LOOP_COUNT*i+50],fa[LOOP_COUNT*i+51],fa[LOOP_COUNT*i+52],fa[LOOP_COUNT*i+53],fa[LOOP_COUNT*i+54],fa[LOOP_COUNT*i+55],
                            fa[LOOP_COUNT*i+56],fa[LOOP_COUNT*i+57],fa[LOOP_COUNT*i+58],fa[LOOP_COUNT*i+59],fa[LOOP_COUNT*i+60],fa[LOOP_COUNT*i+61],fa[LOOP_COUNT*i+62],fa[LOOP_COUNT*i+63]);

    fa4[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i+64],fa[LOOP_COUNT*i+65],fa[LOOP_COUNT*i+66],fa[LOOP_COUNT*i+67],fa[LOOP_COUNT*i+68],fa[LOOP_COUNT*i+69],fa[LOOP_COUNT*i+70],fa[LOOP_COUNT*i+71],
                            fa[LOOP_COUNT*i+71],fa[LOOP_COUNT*i+73],fa[LOOP_COUNT*i+74],fa[LOOP_COUNT*i+75],fa[LOOP_COUNT*i+76],fa[LOOP_COUNT*i+77],fa[LOOP_COUNT*i+78],fa[LOOP_COUNT*i+79]);

    fa5[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i+80],fa[LOOP_COUNT*i+81],fa[LOOP_COUNT*i+82],fa[LOOP_COUNT*i+83],fa[LOOP_COUNT*i+84],fa[LOOP_COUNT*i+85],fa[LOOP_COUNT*i+86],fa[LOOP_COUNT*i+87],
                            fa[LOOP_COUNT*i+88],fa[LOOP_COUNT*i+89],fa[LOOP_COUNT*i+90],fa[LOOP_COUNT*i+91],fa[LOOP_COUNT*i+92],fa[LOOP_COUNT*i+93],fa[LOOP_COUNT*i+94],fa[LOOP_COUNT*i+95]);

    fa6[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i+96],fa[LOOP_COUNT*i+97],fa[LOOP_COUNT*i+98],fa[LOOP_COUNT*i+99],fa[LOOP_COUNT*i+100],fa[LOOP_COUNT*i+101],fa[LOOP_COUNT*i+102],fa[LOOP_COUNT*i+103],
                            fa[LOOP_COUNT*i+104],fa[LOOP_COUNT*i+105],fa[LOOP_COUNT*i+106],fa[LOOP_COUNT*i+107],fa[LOOP_COUNT*i+108],fa[LOOP_COUNT*i+109],fa[LOOP_COUNT*i+110],fa[LOOP_COUNT*i+111]);

    fa7[i].reg = _mm512_set_ps (fa[LOOP_COUNT*i+112],fa[LOOP_COUNT*i+113],fa[LOOP_COUNT*i+114],fa[LOOP_COUNT*i+115],fa[LOOP_COUNT*i+116],fa[LOOP_COUNT*i+117],fa[LOOP_COUNT*i+118],fa[LOOP_COUNT*i+119],
                            fa[LOOP_COUNT*i+120],fa[LOOP_COUNT*i+121],fa[LOOP_COUNT*i+122],fa[LOOP_COUNT*i+123],fa[LOOP_COUNT*i+124],fa[LOOP_COUNT*i+125],fa[LOOP_COUNT*i+126],fa[LOOP_COUNT*i+127]);



    fb0[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i],fb[LOOP_COUNT*i+1],fb[LOOP_COUNT*i+2],fb[LOOP_COUNT*i+3],fb[LOOP_COUNT*i+4],fb[LOOP_COUNT*i+5],fb[LOOP_COUNT*i+6],fb[LOOP_COUNT*i+7],
                            fb[LOOP_COUNT*i+8],fb[LOOP_COUNT*i+9],fb[LOOP_COUNT*i+10],fb[LOOP_COUNT*i+11],fb[LOOP_COUNT*i+12],fb[LOOP_COUNT*i+13],fb[LOOP_COUNT*i+14],fb[LOOP_COUNT*i+15]);

    fb1[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i+16],fb[LOOP_COUNT*i+17],fb[LOOP_COUNT*i+18],fb[LOOP_COUNT*i+19],fb[LOOP_COUNT*i+20],fb[LOOP_COUNT*i+21],fb[LOOP_COUNT*i+22],fb[LOOP_COUNT*i+23],
                            fb[LOOP_COUNT*i+24],fb[LOOP_COUNT*i+25],fb[LOOP_COUNT*i+26],fb[LOOP_COUNT*i+27],fb[LOOP_COUNT*i+28],fb[LOOP_COUNT*i+29],fb[LOOP_COUNT*i+30],fb[LOOP_COUNT*i+31]);

    fb2[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i+32],fb[LOOP_COUNT*i+33],fb[LOOP_COUNT*i+34],fb[LOOP_COUNT*i+35],fb[LOOP_COUNT*i+36],fb[LOOP_COUNT*i+37],fb[LOOP_COUNT*i+38],fb[LOOP_COUNT*i+39],
                            fb[LOOP_COUNT*i+40],fb[LOOP_COUNT*i+41],fb[LOOP_COUNT*i+42],fb[LOOP_COUNT*i+43],fb[LOOP_COUNT*i+44],fb[LOOP_COUNT*i+45],fb[LOOP_COUNT*i+46],fb[LOOP_COUNT*i+47]);

    fb3[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i+48],fb[LOOP_COUNT*i+49],fb[LOOP_COUNT*i+50],fb[LOOP_COUNT*i+51],fb[LOOP_COUNT*i+52],fb[LOOP_COUNT*i+53],fb[LOOP_COUNT*i+54],fb[LOOP_COUNT*i+55],
                            fb[LOOP_COUNT*i+56],fb[LOOP_COUNT*i+57],fb[LOOP_COUNT*i+58],fb[LOOP_COUNT*i+59],fb[LOOP_COUNT*i+60],fb[LOOP_COUNT*i+61],fb[LOOP_COUNT*i+62],fb[LOOP_COUNT*i+63]);

    fb4[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i+64],fb[LOOP_COUNT*i+65],fb[LOOP_COUNT*i+66],fb[LOOP_COUNT*i+67],fb[LOOP_COUNT*i+68],fb[LOOP_COUNT*i+69],fb[LOOP_COUNT*i+70],fb[LOOP_COUNT*i+71],
                            fb[LOOP_COUNT*i+71],fb[LOOP_COUNT*i+73],fb[LOOP_COUNT*i+74],fb[LOOP_COUNT*i+75],fb[LOOP_COUNT*i+76],fb[LOOP_COUNT*i+77],fb[LOOP_COUNT*i+78],fb[LOOP_COUNT*i+79]);

    fb5[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i+80],fb[LOOP_COUNT*i+81],fb[LOOP_COUNT*i+82],fb[LOOP_COUNT*i+83],fb[LOOP_COUNT*i+84],fb[LOOP_COUNT*i+85],fb[LOOP_COUNT*i+86],fb[LOOP_COUNT*i+87],
                            fb[LOOP_COUNT*i+88],fb[LOOP_COUNT*i+89],fb[LOOP_COUNT*i+90],fb[LOOP_COUNT*i+91],fb[LOOP_COUNT*i+92],fb[LOOP_COUNT*i+93],fb[LOOP_COUNT*i+94],fb[LOOP_COUNT*i+95]);

    fb6[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i+96],fb[LOOP_COUNT*i+97],fb[LOOP_COUNT*i+98],fb[LOOP_COUNT*i+99],fb[LOOP_COUNT*i+100],fb[LOOP_COUNT*i+101],fb[LOOP_COUNT*i+102],fb[LOOP_COUNT*i+103],
                            fb[LOOP_COUNT*i+104],fb[LOOP_COUNT*i+105],fb[LOOP_COUNT*i+106],fb[LOOP_COUNT*i+107],fb[LOOP_COUNT*i+108],fb[LOOP_COUNT*i+109],fb[LOOP_COUNT*i+110],fb[LOOP_COUNT*i+111]);

    fb7[i].reg = _mm512_set_ps (fb[LOOP_COUNT*i+112],fb[LOOP_COUNT*i+113],fb[LOOP_COUNT*i+114],fb[LOOP_COUNT*i+115],fb[LOOP_COUNT*i+116],fb[LOOP_COUNT*i+117],fb[LOOP_COUNT*i+118],fb[LOOP_COUNT*i+119],
                            fb[LOOP_COUNT*i+120],fb[LOOP_COUNT*i+121],fb[LOOP_COUNT*i+122],fb[LOOP_COUNT*i+123],fb[LOOP_COUNT*i+124],fb[LOOP_COUNT*i+125],fb[LOOP_COUNT*i+126],fb[LOOP_COUNT*i+127]);
  }
#endif

  struct timeval t0,t1;
  gettimeofday(&t0, 0);
#pragma omp parallel for
  for(t=0; t<N_THREADS; t++)
    compute(t,mult);

  gettimeofday(&t1, 0);
  for(int t=0; t<N_THREADS; t++) {
    fa0[t].reg = INTR4(fa0[t].reg, fa1[t].reg);
    fa2[t].reg = INTR4(fa2[t].reg, fa3[t].reg);
    fa4[t].reg = INTR4(fa4[t].reg, fa5[t].reg);
    fa6[t].reg = INTR4(fa6[t].reg, fa7[t].reg);

    fa0[t].reg = INTR4(fa0[t].reg, fa2[t].reg);
    fa4[t].reg = INTR4(fa4[t].reg, fa6[t].reg);

    fa0[t].reg = INTR4(fa0[t].reg, fa4[t].reg);
  }

  for(int t=1; t<N_THREADS; t++)fa0[0].reg += fa0[t].reg;

  fprintf(stderr,"%f\n\n",sum(fa0[0].reg));

  double e_time = (double)((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec)/1000000;
  double gflops = (double)((long)N_THREADS*MAXFLOPS_ITERS*N_FA_ARRAYS*(BYTES_IN_VECT/4)*2)/1000000000;
  fprintf(stderr, "Used %fs\n",e_time);
  fprintf(stderr, "Computed %.3f GFLOPS\n", gflops);
  fprintf(stderr, "%f GFLOPS/s\n",gflops/e_time);
}
