#include "includes.h"
__global__ void dot(float*a, float*b, float*c, int threadperblock, int max){
__shared__ float cache[ThreadPerBlock];
int tid = threadIdx.x + blockDim.x*blockIdx.x;
float temp = 0;
int cacheindex = threadIdx.x;
while (tid < max){
temp = a[tid] * b[tid];
tid += gridDim.x*blockDim.x;
}
cache[cacheindex] = temp;

__syncthreads();
int i = blockDim.x / 2;
while (i != 0){
if (cacheindex < i)
cache[cacheindex] += cache[cacheindex + i];
__syncthreads();
i /= 2;
}
if (cacheindex == 0)
c[blockIdx.x] = cache[0];



}