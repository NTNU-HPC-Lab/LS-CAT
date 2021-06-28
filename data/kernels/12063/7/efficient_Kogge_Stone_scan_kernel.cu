#include "includes.h"
__global__ void efficient_Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize) {
__shared__ float XY[SECTION_SIZE];
__shared__ float AUS[BLOCK_DIM];
//int i = blockIdx.x * blockDim.x + threadIdx.x;

// Keep mind: Partition the input into blockDim.x subsections: i.e. for 8 threads --> 8 subsections

// collaborative load in a coalesced manner
for (int j = 0; j < SECTION_SIZE; j += blockDim.x) {
XY[threadIdx.x + j] = X[threadIdx.x + j];
}
__syncthreads();


// PHASE 1: scan inner own subsection
// At the end of this phase the last element of each subsection contains the sum of all alements in own subsection
for (int j = 1; j < SUBSECTION_SIZE; j++) {
XY[threadIdx.x * (SUBSECTION_SIZE)+j] += XY[threadIdx.x * (SUBSECTION_SIZE)+j - 1];
}
__syncthreads();


// PHASE 2: perform iterative kogge_stone_scan of the last elements of each subsections of XY loaded first in AUS
AUS[threadIdx.x] = XY[threadIdx.x * (SUBSECTION_SIZE)+(SUBSECTION_SIZE)-1];
float in;
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
__syncthreads();
if (threadIdx.x >= stride) {
in = AUS[threadIdx.x - stride];
}
__syncthreads();
if (threadIdx.x >= stride) {
AUS[threadIdx.x] += in;
}
}
__syncthreads();


// PHASE 3: each thread adds to its elements the new value of the last element of its predecessor's section
if (threadIdx.x > 0) {
for (unsigned int stride = 0; stride < (SUBSECTION_SIZE); stride++) {
XY[threadIdx.x * (SUBSECTION_SIZE)+stride] += AUS[threadIdx.x - 1];  // <--
}
}
__syncthreads();


// store the result into output vector
for (int j = 0; j < SECTION_SIZE; j += blockDim.x) {
Y[threadIdx.x + j] = XY[threadIdx.x + j];
}
}