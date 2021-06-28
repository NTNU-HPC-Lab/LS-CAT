#include "includes.h"
__global__ void matadd(const float *a, const float *b, float *c, int n, int m){
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;
int idx = j * n + i;
if(i < n and j < m){
c[idx] = a[idx] + b[idx];
}
}