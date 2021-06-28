#include "includes.h"
__global__ void uniform_float(int n,float lower,float upper,float *result) {
int totalThreads = gridDim.x * blockDim.x;
int tid = threadIdx.x;
int i = blockIdx.x * blockDim.x + tid;

for(; i < n; i += totalThreads) {
float u = result[i];
result[i] = u * upper + (1 - u) * lower;
}
}