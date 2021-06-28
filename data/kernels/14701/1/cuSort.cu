#include "includes.h"
__global__ void cuSort(float* data,int bucketSize,int* startPoint)
{

//	int L= blockIdx.x * blockDim.x;
int L= blockIdx.x*bucketSize;
int U= L + bucketSize;
int j;
float tmp;
startPoint[blockIdx.x] = L;
for(int i=L+1; i < U; i++)
{
tmp=data[i];
j = i-1;
while(tmp<data[j] && j>=0)
{
data[j+1] = data[j];
j = j-1;
}
data[j+1]=tmp;
}
}