#include "includes.h"
__global__ void kernel_hadamard_fl(int N, float *wt, float *x){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only M threads */
if (tid<N) {
x[tid]*=wt[tid];
}
}