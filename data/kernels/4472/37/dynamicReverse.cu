#include "includes.h"
__global__ void dynamicReverse(int *d, int n)
{
extern __shared__ int s[];
int t = threadIdx.x;
int tr = n-t-1;
s[t] = d[t];
__syncthreads();
d[t] = s[tr];
}