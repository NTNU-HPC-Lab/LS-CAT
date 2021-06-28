#include "includes.h"






__global__ void reduce_fields(float *d_rho, float *d_Ex, float* d_Ey, float* d_Ez, float *d_Rrho, float* d_REx, float* d_REy, float* d_REz, int N)
{
__shared__ float rho_array[gThreadsAll];
__shared__ float Ex_array[gThreadsAll];
__shared__ float Ey_array[gThreadsAll];
__shared__ float Ez_array[gThreadsAll];
int n = blockDim.x * blockIdx.x + threadIdx.x;
if (n < N){
for (int s = blockDim.x / 2; s > 0; s >>= 1){
if ( threadIdx.x < s)
{
rho_array[threadIdx.x] += d_rho[threadIdx.x + s];
Ex_array[threadIdx.x] += d_Ex[threadIdx.x + s] * d_Ex[threadIdx.x + s];
Ey_array[threadIdx.x] += d_Ey[threadIdx.x + s] * d_Ey[threadIdx.x + s];
Ez_array[threadIdx.x] += d_Ez[threadIdx.x + s] * d_Ez[threadIdx.x + s];
}
__syncthreads();
}

if (threadIdx.x ==0){
d_Rrho[blockIdx.x] = rho_array[0];
d_REx[blockIdx.x] = Ex_array[0];
d_REy[blockIdx.x] = Ey_array[0];
d_REz[blockIdx.x] = Ez_array[0];
}
}
}