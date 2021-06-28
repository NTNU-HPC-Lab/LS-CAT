#include "includes.h"
//=============================================================================
// FILE:   mytoy.cu
// AUTHORS: Raul Segura & Manuel Ujaldon (copyright 2014)
// Look for the string "MU" whenever Manuel suggests you to introduce changes
// Feel free to change some other parts of the code too (at your own risk)
//=============================================================================



//=============================================================================
// CUDA functions.
//=============================================================================

//Error handler for CUDA functions.
__global__ void kernelAdd(float *dvalues, int numOperations, int firstInd, int nextColInd)
{
int vi = firstInd + blockIdx.x * blockDim.x + threadIdx.x;

// "numOperations" is the 2nd input parameter to our executable
if (vi < nextColInd) {
for (int j=0; j<numOperations; ++j) {
// The operation performed on each nonzero of our sparse matrix:
dvalues[vi] *=dvalues[vi]+dvalues[vi]*dvalues[vi]; // POINT 3: Choices you may try here:
}                               // *= (for multiply), /= (for division),
}                                   // or you may investigate some other :-)
}