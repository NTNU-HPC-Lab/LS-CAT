#include "includes.h"
__global__ void PowerInterleaved(float4 *src, float4 *dest) {

const size_t i = blockDim.x * blockIdx.x + threadIdx.x;

// Cross pols
dest[i].x  += src[i].x * src[i].x + src[i].y * src[i].y;
dest[i].y  += src[i].z * src[i].z + src[i].w * src[i].w;
// Parallel pols
dest[i].z += src[i].x * src[i].z + src[i].y * src[i].w;
dest[i].w += src[i].y * src[i].z - src[i].x * src[i].w;
}