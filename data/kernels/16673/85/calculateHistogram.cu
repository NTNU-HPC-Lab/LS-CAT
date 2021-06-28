#include "includes.h"
__global__ void calculateHistogram(unsigned int *imageHistogram, unsigned int width, unsigned int height, cudaTextureObject_t texObj)
{
const unsigned int tidX = blockIdx.x*blockDim.x + threadIdx.x;
const unsigned int tidY = blockIdx.y*blockDim.y + threadIdx.y;

const unsigned int localId = threadIdx.y*blockDim.x+threadIdx.x;
const unsigned int histStartIndex = (blockIdx.y*gridDim.x+blockIdx.x) * 256;

__shared__ unsigned int histo_private[256];

if(localId <256)
histo_private[localId] = 0;
__syncthreads();

// Step 4: Read the texture memory from your texture reference in CUDA Kernel
unsigned char imageData =  tex2D<unsigned char>(texObj,(float)(tidX),(float)(tidY));
atomicAdd(&(histo_private[imageData]), 1);

__syncthreads();

if(localId < 256)
imageHistogram[histStartIndex+localId] = histo_private[localId];

}