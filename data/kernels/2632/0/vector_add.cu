#include "includes.h"
__global__ void vector_add(float *out, float *a, float *b, int n) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for(int i = index; i < n; i += stride){
out[i] = a[i] + b[i];
}
}