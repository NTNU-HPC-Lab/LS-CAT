#include "includes.h"
__global__ static void findRHS(double* cOld, double* cCurr, double* cHalf, double* cNonLinRHS, int nx)
{
// Matrix index
int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

// Set index being computed
int index = globalIdy * nx + globalIdx;

// Set the RHS for inversion
cHalf[index] += - (2.0 / 3.0) * (cCurr[index] - cOld[index]) + cNonLinRHS[index];

// Set cOld to cCurr
cOld[index] = cCurr[index];
}