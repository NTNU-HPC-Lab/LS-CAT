#include "includes.h"
__global__ void var(int *a,int *b,int n,float mean)
{


int block=256*blockIdx.x;
float sum=0;


for(int i=block;i<min(block+256,n);i++)
{


sum=sum+(a[i]-mean)*(a[i]-mean);


}
b[blockIdx.x]=sum;

}