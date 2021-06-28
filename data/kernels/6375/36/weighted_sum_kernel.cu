#include "includes.h"
__global__ void weighted_sum_kernel(int n, float *a, float *b, float *s, float *c)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
}
}