#include "includes.h"
__global__ void kernel_hadamard_sum(int N, double *y, double *x, double *w){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only N threads */
if (tid<N) {
y[tid]+=x[tid]*w[tid];
}
}