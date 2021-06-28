#include "includes.h"

#define tileSize 32

//function for data initialization
void initialization( double *M,  double *N, int arow, int acol, int brow, int bcol);
//(for Debugging) prints out the input data
void printInput( double *M,  double *N, int arow, int acol, int brow,  int bcol);
//(for Debugging) prints out the output data
void printOutput( double *P_C,  double *P_G, int arow, int bcol);

//GPU kernels




__global__
__global__ void vectorAddition(const double *A, const double *B, double *C, int numElements)
{
int gridIndex = blockDim.x * blockIdx.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;

for (int i = gridIndex; i<numElements; i+=stride)
{
C[i] = A[i] + B[i];
}
}