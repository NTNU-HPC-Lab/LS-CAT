#include "includes.h"
__global__ void transpose(int N, double *A)
{
int row,col,k;
double temp;
k = (blockIdx.y*gridDim.x+blockIdx.x)*(blockDim.x*blockDim.y)+(threadIdx.y*blockDim.x+threadIdx.x);
row = k/N;
col = k - row*N;
if(row<col){
temp = A[row*N+col];
A[row*N+col] = A[col*N+row];
A[col*N+row] = temp;
}
}