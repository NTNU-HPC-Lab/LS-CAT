#include "includes.h"
__global__ void sum(int *a,int *b,int n)
{
int block=256*blockIdx.x;
int sum=0;
for(int i=block;i<min(block+256,n);i++)
{
sum=sum+a[i];
}
b[blockIdx.x]=sum;
}