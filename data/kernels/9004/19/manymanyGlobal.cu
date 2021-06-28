#include "includes.h"
__global__ void manymanyGlobal(int* a,int* b)
{

for(int j=0; j < ITER; j++)
for(int i=threadIdx.x;i<SIZE;i+=THREAD)
{
a[i]=0;
b[i]=0;
}
}