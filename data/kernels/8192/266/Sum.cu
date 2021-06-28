#include "includes.h"
__global__ void Sum(float * A, float  *B, float *C, int size) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
if (id < size) {
C[id] = A[id] + B[id];
}
}