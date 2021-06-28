#include "includes.h"
__global__ void cudaFillArray( float *gpu_array, float val, int N )
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if( i < N ){
gpu_array[i] = val;
}
}