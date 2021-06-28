#include "includes.h"
__global__ void uniform_double(int n,double lower,double upper,double *result) {
int totalThreads = gridDim.x * blockDim.x;
int tid = threadIdx.x;
int i = blockIdx.x * blockDim.x + tid;

for(; i < n; i += totalThreads) {
double u = result[i];
result[i] = u * upper + (1 - u) * lower;
}
}