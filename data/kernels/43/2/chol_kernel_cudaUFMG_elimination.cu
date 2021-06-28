#include "includes.h"
__global__ void chol_kernel_cudaUFMG_elimination(float * U, int k) {

//This call acts as a single K iteration
//Each block does a single i iteration
//Need to consider offset,
int i = (k+1) + blockIdx.x;

//Each thread does some part of j
//Stide in units of 'stride'
//Thread 0 does 0, 16, 32
//Thread 1 does 1, 17, 33
//..etc.
int jstart = i + threadIdx.x;
int jstep = blockDim.x;

// Pre-calculate indexes
int kM = k * MATRIX_SIZE;
int iM = i * MATRIX_SIZE;
int ki = kM + i;

//Do work for this i iteration
//Want to stride across
for (int j=jstart; j<MATRIX_SIZE; j+=jstep) {
U[iM + j] -= U[ki] * U[kM + j];
}
}