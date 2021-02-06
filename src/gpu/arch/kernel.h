#ifndef __KERNEL__
#define __KERNEL__

#define KERNEL_ITERS 100000000

struct benchmark_gpu;

struct benchmark_gpu* init_benchmark_gpu();
double get_gflops_gpu(struct benchmark_gpu* bench);
bool compute_gpu(struct benchmark_gpu* bench);
const char* get_benchmark_name_gpu(struct benchmark_gpu* bench);
void exit_benchmark_gpu();

#endif
