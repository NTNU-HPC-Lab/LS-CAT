#include "includes.h"
__global__ void _copy_mat(float *m, float* target, int len){
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid < len){
target[tid] = m[tid];
}
}