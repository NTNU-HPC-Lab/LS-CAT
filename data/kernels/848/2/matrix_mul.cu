#include "includes.h"
__global__ void matrix_mul(float *ad,float *bd,float *cd,int N)
{
float pvalue=0;

//find Row and Column corresponding to a data element for each thread
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;

//calculate dot product of Row of First Matrix and Column of Second Matrix
for(int i=0;i< N;++i)
{
float m=ad[Row * N+i];
float n=bd[i * N + Col];
pvalue += m * n;
}

//store dot product at corresponding positon in resultant Matrix
cd[Row * N + Col] = pvalue;

}