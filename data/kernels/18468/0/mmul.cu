#include "includes.h"

// these are just for timing measurments

// error checking macro
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

if ((idx < ds) && (idy < ds)){
float temp = 0;
for (int i = 0; i < ds; i++)
temp += A[idy*ds+i] * B[i*ds+idx];   // dot product of row and column
C[idy*ds+idx] = temp;
}
}