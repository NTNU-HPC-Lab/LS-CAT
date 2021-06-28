#include "includes.h"
__global__ void matrix_transpose_k1(float* input,float* output,const int nx, const int ny)
{
int gid = blockDim.x * blockIdx.x + threadIdx.x;
int offset = threadIdx.x*blockDim.x;
//printf("gid : %d , offset : %d , index : %d ,value : %f \n", gid, offset, offset + blockIdx.x,input[offset + blockIdx.x]);
output[gid] = input[offset + blockIdx.x];
}