#include "includes.h"
__global__ void matmul(const float *a, const float *b, float *c, int n, int m){
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;
//printf("%d %d %d %d %d %d\n",blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
int idx = j * n + i;
if(i < n and j < m){
//printf("%d %d %d %d %d %d\n", i, j, idx, a[idx], b[idx], c[idx]);
float sum = 0;
for(int k = 0; k < n; k++){
int idxa = j * n + k;
int idxb = k * n + i;
sum += a[idxa] * b[idxb];
}
c[idx] = sum;
}
}