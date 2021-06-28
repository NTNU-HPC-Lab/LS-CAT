#include "includes.h"
__global__ void kernMoveMem(const size_t numPoints, const size_t pointDim, const size_t s, double* A) {
int b = blockIdx.y * gridDim.x + blockIdx.x;
int i = b * blockDim.x + threadIdx.x;

// Before
// [abc......] [def......] [ghi......] [jkl......]

// shared memory
// [adgj.....]

// After
// [a..d..g..] [j........] [ghi......] [.........]

__shared__ double mem[1024];
mem[threadIdx.x] = A[s * i * pointDim];
__syncthreads();
A[i * pointDim] = mem[threadIdx.x];
}