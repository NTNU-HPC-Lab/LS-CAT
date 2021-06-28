#include "includes.h"
__global__ void resized(unsigned char *imgData, int width, float scale_factor, cudaTextureObject_t texObj) {
const unsigned  int tidX = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned  int tidY = blockIdx.y * blockDim.y + threadIdx.y;

const unsigned idx = tidY * width + tidX;

//Read texture mem to CUDA Kernel

imgData[idx] = tex2D<unsigned char>(texObj,(float)(tidX*scale_factor),(float)(tidY*scale_factor));

}