#include "includes.h"
__global__ void hierarchical_scan_kernel_phase1(int *X, int *Y, int *S) {
__shared__ int XY[SECTION_SIZE];
__shared__ int AUS[BLOCK_DIM];
int tx = threadIdx.x, bx = blockIdx.x;
int i = bx * SECTION_SIZE + tx;

if (i < INPUT_SIZE) {

// collaborative load in a coalesced manner
for (int j = 0; j < SECTION_SIZE; j+=BLOCK_DIM) {
XY[tx + j] = X[i + j];
}
__syncthreads();


// PHASE 1: scan inner own subsection
// At the end of this phase the last element of each subsection contains the sum of all alements in own subsection
for (int j = 1; j < SUBSECTION_SIZE; j++) {
XY[tx * (SUBSECTION_SIZE) + j] += XY[tx * (SUBSECTION_SIZE)+j - 1];
}
__syncthreads();


// PHASE 2: perform iterative kogge_stone_scan of the last elements of each subsections of XY loaded first in AUS
AUS[tx] = XY[tx * (SUBSECTION_SIZE)+(SUBSECTION_SIZE)-1];
int in;
for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2) {
__syncthreads();
if (tx >= stride) {
in = AUS[tx - stride];
}
__syncthreads();
if (tx >= stride) {
AUS[tx] += in;
}
}
__syncthreads();

// PHASE 3: each thread adds to its elements the new value of the last element of its predecessor's section
if (tx > 0) {
for (unsigned int stride = 0; stride < (SUBSECTION_SIZE); stride++) {
XY[tx * (SUBSECTION_SIZE)+stride] += AUS[tx - 1];  // <--
}
}
__syncthreads();

// store the result into output vector
for (int j = 0; j < SECTION_SIZE; j += BLOCK_DIM) {
Y[i + j] = XY[tx + j];
}

//The last thread in the block writes the output value of the last element in the scan block to the blockIdx.x position of S
if (tx == BLOCK_DIM - 1) {
S[bx] = XY[SECTION_SIZE - 1];
}
}
}