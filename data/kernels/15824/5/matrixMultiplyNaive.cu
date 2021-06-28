#include "includes.h"
__global__ void matrixMultiplyNaive(float * A, float * B, float * C, int N,int K,int M)
{

int Row = blockDim.y*blockIdx.y + threadIdx.y; //To generate ids of threads.
int Col = blockDim.x*blockIdx.x + threadIdx.x;

if(Row<N && Col<M)
{
float Cvalue = 0.0;
int k;
for(k=0;k<K;k++)
{
Cvalue += A[Row*K+k] * B[k*M+Col];
}
C[Row*M+Col] = Cvalue;
}
}