#include "includes.h"
__global__ void _init_mat(float *m, float val, int len){
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid < len){
m[tid] = val;
}
}