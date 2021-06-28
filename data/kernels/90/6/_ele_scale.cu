#include "includes.h"
__global__ void _ele_scale(float *m, float *target, float scaler, int len){
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid < len){
target[tid] = scaler * m[tid];
}
}