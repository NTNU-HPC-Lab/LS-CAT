#include "includes.h"
__global__ void kernel_hadamard(int N, double *wt, double *x){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only N threads */
if (tid<N) {
x[tid]*=wt[tid];
}
}