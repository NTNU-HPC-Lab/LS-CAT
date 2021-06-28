#include "includes.h"
__global__ void cal_hist(float *da, int *hist_da, int N, int M){
int bx = blockIdx.x;
int tx = threadIdx.x;
int idx = bx * blockDim.x + tx;
if(idx < N){
// add a lock here to make sure this (read, write) operation atomic.
atomicAdd(&hist_da[(int)da[idx]], 1);
//hist_da[(int)da[idx]] += 1;
}
}