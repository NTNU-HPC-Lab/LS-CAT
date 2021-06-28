#include "includes.h"
__global__ void xmpC2S_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out) {
//outer dimension = N
//inner dimension = limbs

//read strided in inner dimension`
//write coalesced in outer dimension
for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
for(uint32_t j=blockIdx.y*blockDim.y+threadIdx.y;j<limbs;j+=blockDim.y*gridDim.y) {
out[j*stride + i] = in[i*limbs + j];
}
}
}