#include "includes.h"
__global__ void Matrix_Product (double *A, double *g, double *C)
// Each thread computes one element of C
// by accumulating results into Cvalue
{               double Cvalue = 0.00;
int row = blockIdx.y*blockDim.y+threadIdx.y;
// int col = blockIdx.x * blockDim.x + threadIdx.x;
//size of matrix A//
int N=1000;
if(row> N ) return;
for (int e = 0; e < N; e++)
{
Cvalue += A[N*row+e]*g[e];
}
C[row]+= Cvalue;
}