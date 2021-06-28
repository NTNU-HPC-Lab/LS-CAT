#include "includes.h"
__global__ void montecarlo(float* d_out, float __lowx, float __highx, float __lowy, float __highy, int __iters) {
__shared__ float lowx, highx, lowy, highy;
__shared__ int iters;

int tid = blockIdx.x * blockDim.x + threadIdx.x;

// let's fix the shared variables for all threads per block once (check the synchronization call).
if (threadIdx.x == 0) {
lowx = __lowx, highx = __highx, lowy = __lowy, highy = __highy;
iters = __iters;
}
__syncthreads();

curandState localState;
curand_init(tid, 0, 0, &localState);

int i;
float x, y, tempSum = 0.;
for (i = 0; i < iters; i ++) { // each thread calculates its own summation.
x = lowx + curand_uniform(&localState) * (highx - lowx);
y = lowy + curand_uniform(&localState) * (highy - lowy);
tempSum += exp(-x * x - y * y);
}
d_out[tid] = tempSum;
}