#include "includes.h"
__global__ void kernel(float *a, size_t N)
{
int tid = threadIdx.x;
__shared__ float s[BS];
int blocks = (N+BS-1)/BS;
float sum = 0.0f;
for (int ib=0; ib<blocks; ib++)
{
int off = ib*BS+tid;
s[tid] = a[off];
for (int skip=16; skip>0; skip>>=1)
if (tid+skip < N && tid < skip)
s[tid] += s[tid+skip];
sum += s[0];
}
a[0] = sum;
}