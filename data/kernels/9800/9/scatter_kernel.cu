#include "includes.h"
__global__ void scatter_kernel(unsigned int* d_inputVals, unsigned int* d_inputPos, unsigned int* d_outputVals, unsigned int* d_outputPos, unsigned int* cu_outputVals, size_t numElems) {
//unsigned int tid = threadIdx.x;
unsigned int mid = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int val;
if (mid < numElems) {
val = cu_outputVals[mid];
}

if (mid < numElems) {
d_outputVals[val] = d_inputVals[mid];
d_outputPos[val] = d_inputPos[mid];
}
__syncthreads();
}