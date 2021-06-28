#include "includes.h"
__global__ static void findCBar(double* cOld, double* cCurr, double* cBar, int nx)
{
// Matrix index
int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

// Set index being computed
int index = globalIdy * nx + globalIdx;

// Find cBar
cBar[index] = 2.0 * cCurr[index] - cOld[index];
}