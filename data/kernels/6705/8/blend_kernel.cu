#include "includes.h"
__global__ void blend_kernel( float *outSrc, const float *inSrc ) {
// map from threadIdx/blockIdx to pixel position
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

int left = offset - 1;
int right = offset + 1;
if(x == 0) left++;
if(x == DIM-1) right--;

int top = offset - DIM;
int bottom = offset + DIM;
if(y == 0) top += DIM;
if(y == DIM-1) bottom -= DIM;

outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
}