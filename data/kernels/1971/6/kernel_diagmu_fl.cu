#include "includes.h"
__global__ void kernel_diagmu_fl(int M, float *A,float mu){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only M threads */
if (tid<M) {
A[tid*(M+1)]=A[tid*(M+1)]+mu;
}
}