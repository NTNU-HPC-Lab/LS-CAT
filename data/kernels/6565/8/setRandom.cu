#include "includes.h"
__device__ unsigned int Rand(unsigned int randx)
{
randx = randx*1103515245+12345;
return randx&2147483647;
}
__global__ void setRandom(float *gpu_array, int N, int maxval )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if( id < N ){
gpu_array[id] = 1.0f / maxval * Rand(id) / float( RAND_MAX );
}
}