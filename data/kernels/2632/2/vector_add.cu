#include "includes.h"
__global__ void vector_add(float *out, float *a, float *b, int n) {
int index = threadIdx.x;
int stride = blockDim.x;
for(int i = index; i < n; i += stride){
out[i] = a[i] + b[i];
}
}