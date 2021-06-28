#include "includes.h"
__global__ void vevAdd(int N, float *a, float *b, float *c)
{
// work idex, 在launch kernel的时候指定维度
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < N)
{
c[idx] = a[idx] + b[idx];
}
}