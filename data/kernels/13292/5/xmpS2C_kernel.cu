#include "includes.h"
__global__ void xmpS2C_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out) {
//outer dimension = limbs
//inner dimension = N

//read strided in inner dimension
//write coalesced in outer dimension
for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<limbs;i+=blockDim.x*gridDim.x) {
for(uint32_t j=blockIdx.y*blockDim.y+threadIdx.y;j<N;j+=blockDim.y*gridDim.y) {
out[j*limbs + i] = in[i*stride + j];
}
}
}