#include "includes.h"
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

// declare cache in shared memory
__shared__ float As[block_size][block_size];
__shared__ float Bs[block_size][block_size];

int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

if ((idx < ds) && (idy < ds)){
float temp = 0;
for (int i = 0; i < ds/block_size; i++) {

// Load data into shared memory
As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

// Synchronize
__syncthreads();

// Keep track of the running sum
for (int k = 0; k < block_size; k++)
temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
__syncthreads();

}

// Write to global memory
C[idy*ds+idx] = temp;
}
}