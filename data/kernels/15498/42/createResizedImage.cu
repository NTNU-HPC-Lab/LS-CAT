#include "includes.h"
__global__ void createResizedImage(unsigned char *imageScaledData, int scaled_width, float scale_factor, cudaTextureObject_t texObj)
{
const unsigned int tidX = blockIdx.x*blockDim.x + threadIdx.x;
const unsigned int tidY = blockIdx.y*blockDim.y + threadIdx.y;
const unsigned index = tidY*scaled_width+tidX;

//Step 3: Read the texture memory from your texture reference in CUDA Kernel
imageScaledData[index] = tex2D<unsigned char>(texObj,(float)(tidX*scale_factor),(float)(tidY*scale_factor));
}