#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void matrixMultiplicationKernel(float *A, float* B, float* C, int a, int b, int d) {

// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

int ROW = by*blockDim.y+ty;
int COL = bx*blockDim.x+tx;

// First check if the thread exceeds the matrix dimensions
if (ROW < a && COL < d) {

// Declaration of the shared memory array As used to store the sub-
// matrix of A
__shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
__shared__ float As2[BLOCK_SIZE * BLOCK_SIZE];

float *prefetch = As;
float *prefetch2 = As2;

// Declaration of the shared memory array Bs used to
// store the sub-matrix of B
// __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

float cv[BLOCK_SIZE];

for (int ii = 0; ii < BLOCK_SIZE; ii++) {
cv[ii] = 0;
}

// Index of the first sub-matrix of A processed by the block
int aBegin = a * BLOCK_SIZE * by;

// Index of the last sub-matrix of A processed by the block
int aEnd   = aBegin + a - 1;

// Step size used to iterate through the sub-matrices of A
int aStep  = BLOCK_SIZE;

// Index of the first sub-matrix of B processed by the block
int bBegin = BLOCK_SIZE * VECTOR_SIZE * bx;

// Step size used to iterate through the sub-matrices of B
int bStep  = BLOCK_SIZE * d;

int cBegin = d * BLOCK_SIZE * by + VECTOR_SIZE * BLOCK_SIZE * bx;

// Csub is used to store the element of the block sub-matrix
// that is computed by the thread
// float Csub = 0;
float *Ap = &A[aBegin + a * ty +tx];
float *ap = &prefetch[ty + BLOCK_SIZE * tx];
#pragma unroll
for(int ii = 0; ii < BLOCK_SIZE; ii+=4){
ap[ii] = Ap[a * ii];
}
__syncthreads();

// Loop over all the sub-matrices of A and B
// required to compute the block sub-matrix
for (int a = aBegin, b = bBegin;
a <= aEnd;
a += aStep, b += bStep) {

// Load the matrices from device memory
// to shared memory; each thread loads
// one element of each matrix
Ap = &A[a + aStep + a * ty +tx];
float *ap2 = &prefetch2[ty + BLOCK_SIZE * tx];
#pragma unroll
for(int ii = 0; ii < BLOCK_SIZE; ii+=4){
ap2[ii] = Ap[b * ii];
}

ap = &prefetch[0];
float *bp = &B[b + BLOCK_SIZE * ty + tx];

#pragma unroll
for (int ii = 0; ii < BLOCK_SIZE; ii++) {
float bv = bp[0];
for (int jj = 0; jj < BLOCK_SIZE; jj++) {
cv[jj] += ap[jj]*bv;
ap += BLOCK_SIZE;
bp += d;
}
}

// Synchronize to make sure the matrices are loaded
__syncthreads();

// swap As and As2
float *prefetch_temp = prefetch;
prefetch = prefetch2;
prefetch2 = prefetch_temp;
}

// Write the block sub-matrix to device memory;
// each thread writes one element
float *Cp = &C[cBegin];
Cp += BLOCK_SIZE * ty + tx;
int cStep = d;
#pragma unroll
for(int ii=0; ii<BLOCK_SIZE; ii++){
Cp[0] = cv[ii]; Cp += cStep;
}
}
}