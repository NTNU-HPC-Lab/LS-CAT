#include "includes.h"
__global__ void Accumulate(float4 *src, float4 *dest, int loop) {
const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
const size_t k = blockDim.x * gridDim.x;

dest[i] = src[i];

for (int n=1; n<loop; n++) {
dest[i].x  += src[i+n*k].x;
dest[i].y  += src[i+n*k].y;
dest[i].z  += src[i+n*k].z;
dest[i].w  += src[i+n*k].w;
}
}