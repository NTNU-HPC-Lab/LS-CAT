#include "includes.h"
__global__ void _mat_add(float *ma, float *mb, float *target, float sa, float sb, int len){
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid < len){
target[tid] = sa * ma[tid] + sb * mb[tid];
}
}