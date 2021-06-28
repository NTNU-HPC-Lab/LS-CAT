#include "includes.h"

#define SIZE 1024


__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
int i = threadIdx.x;

if(i<n)
c[i]=a[i]+b[i];
}