#include "includes.h"
__global__ void matrixAdd(int *a,int *b,int *c)
{
int col=blockIdx.x*blockDim.x+threadIdx.x;
int row=blockIdx.y*blockDim.y+threadIdx.y;
int index=col+row*N;
printf("\n%d\t%d",threadIdx.x,threadIdx.y);
printf("\nIndex val:%d\n",index);
if(col<N && row<N)
{
c[index]=a[index]+b[index];
}
}