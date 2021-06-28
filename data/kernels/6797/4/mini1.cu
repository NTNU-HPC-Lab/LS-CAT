#include "includes.h"
__global__ void mini1(int *a,int *b,int n)
{


int block=256*blockIdx.x;


int mini=7888888;

for(int i=block;i<min(256+block,n);i++)
{


if(mini>a[i])
{

mini=a[i];

}





}
b[blockIdx.x]=mini;

}