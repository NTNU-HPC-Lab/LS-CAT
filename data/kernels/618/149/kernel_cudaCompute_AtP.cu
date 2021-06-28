#include "includes.h"
__global__ void kernel_cudaCompute_AtP(double *d_A, double *d_P, double *d_AtP, int rows, int columns )
{
int ind=blockIdx.x*blockDim.x+threadIdx.x;
if(ind<rows*columns)
{
int row = ind%rows;
int column = ind/rows;

d_AtP[row + column * rows] = d_A[column + row * columns] * d_P[column];
}
}