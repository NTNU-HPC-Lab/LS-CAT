#include "includes.h"
__global__ void matmul_traditional(const float *a, const float *b, float *c, int n, int m){
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;
//printf("%d %d %d %d %d %d\n",blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
int idx = i * n + j;


int2 i2 = make_int2(1, 2);
float4 f4 = make_float4(0, 0, 0, 0);
f4.x = 0.1, f4.y = 0.2, f4.z = 0.3, f4.w = 0.4;
//printf("%d %d %f %f %f %f\n", i2.x, i2.y, f4.x, f4.y, f4.z, f4.w);

if(i < n and j < m){
//printf("%d %d %d %d %d %d\n", i, j, idx, a[idx], b[idx], c[idx]);
float sum = 0;
for(int k = 0; k < n; k++){
int idxa = i * n + k;
int idxb = k * n + j;
sum += a[idxa] * b[idxb];
}
c[idx] = sum;
}
}