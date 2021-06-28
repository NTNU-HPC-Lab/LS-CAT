#include "includes.h"
__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

/********************************************************************
*
* Compute C = A x B
*   where A is a (m x k) matrix
*   where B is a (k x n) matrix
*   where C is a (m x n) matrix
*
********************************************************************/

// INSERT KERNEL CODE HERE
int row, col;

row = blockIdx.y*blockDim.y+threadIdx.y;

col = blockIdx.x*blockDim.x+threadIdx.x;


if(( row < m) && (col < n))
{
float acc = 0;

for(int index = 0; index < k; index++)
{
acc = acc + A[row * k + index] * B[index * n + col];
}

C[row * n + col] = acc;

}
}