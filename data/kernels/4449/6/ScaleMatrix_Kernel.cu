#include "includes.h"
__global__ void ScaleMatrix_Kernel(float *d_a, float alpha, int arraySize)
{
// Block index
int bx = blockIdx.x;

// Thread index
int tx = threadIdx.x;
int begin = blockDim.x * bx;
int index = begin + tx;

// copies array into shared memory, important only if threads are communicating between each other. Its not necessary here since we are only scaling vector.

__shared__ float d_as[BLOCKSIZE];

d_as[tx] = d_a[index];

__syncthreads();

// copies array back to global device memory

d_a[index] = alpha * d_as[tx];

}