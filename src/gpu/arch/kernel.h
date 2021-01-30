#ifndef __KERNEL__
#define __KERNEL__

#define KERNEL_ITERS 100000000

struct benchmark;

struct benchmark* init_benchmark();
int get_n(struct benchmark* bench);
double get_tflops(struct benchmark* bench);
void compute(struct benchmark* bench, float *a, float *b, float *c, int n);
void print_benchmark(struct benchmark* bench);

#endif
