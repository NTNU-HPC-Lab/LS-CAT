#include "includes.h"
__global__ void _mat_mul(float *ma, float *mb, float *target, int len){
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid < len){
target[tid] = ma[tid] * mb[tid];
}
}