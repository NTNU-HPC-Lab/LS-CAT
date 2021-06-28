#include "includes.h"
__global__ void copy_const_kernel ( float *iptr, const float *cptr ) {
// map from threadIdx/blockIdx to pixel position
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

if(cptr[offset] != 0) iptr[offset] = cptr[offset];
}