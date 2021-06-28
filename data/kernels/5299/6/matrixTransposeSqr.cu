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
__global__ void matrixTransposeSqr(double *P, double* M, int width, int height)
{
unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

if (xIdx < width && yIdx < height)
{
unsigned int inIdx  = xIdx + width * yIdx;
unsigned int outIdx= yIdx + height * xIdx;
P[outIdx] = M[inIdx];
}
}