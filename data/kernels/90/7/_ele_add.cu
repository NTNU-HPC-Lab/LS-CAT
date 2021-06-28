#include "includes.h"
__global__ void _ele_add(float *m, float *target, float val, int len){
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid < len){
target[tid] = val + m[tid];
}
}