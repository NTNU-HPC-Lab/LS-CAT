#include "includes.h"
__global__ static void findNew(double* cCurr, double* cBar, double* cHalf, int nx)
{
// Matrix index
int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

// Set index being computed
int index = globalIdy * nx + globalIdx;

// Recover the new data
cCurr[index] = cBar[index] + cHalf[index];
}