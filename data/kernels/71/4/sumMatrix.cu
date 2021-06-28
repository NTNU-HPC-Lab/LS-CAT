#include "includes.h"
__global__ void sumMatrix(float *A, float *B, float *C, int nx, int ny) {
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = ix + iy * nx;
if(ix < nx && iy < ny) {
C[idx] = A[idx] + B[idx];
}
}