#include "includes.h"
__global__ void matadd_2d(const float *a, const float *b, float *c, int n, int m){
int i =  blockDim.x * blockIdx.x + threadIdx.x;
int j =  blockIdx.y;
if(i < n and j < m){
int idx = j * n + i;
c[idx] = a[idx] + b[idx];
}
}