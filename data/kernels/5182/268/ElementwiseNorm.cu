#include "includes.h"
__global__ void ElementwiseNorm(float * A, float  *B, int size) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
if (id < size)
A[id] /= B[id];
}