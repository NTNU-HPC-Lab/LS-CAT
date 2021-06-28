#include "includes.h"
__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
for(long i=0;i<1024*500;i++){
c[i] = a[i]*10 + b[i] * 5;
}
//printf("addKernel::threadIdx: %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}