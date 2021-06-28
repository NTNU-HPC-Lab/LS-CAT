#include "includes.h"
__global__ void useSingleTexture(cudaTextureObject_t tex, float* pout)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

float4 sample = tex3D<float4>(tex, i + 0.5, j + 0.5, k + 0.5);

pout[i + c_size.x * (j + k * c_size.y)] = sqrtf(powf(sample.x,2)+ powf(sample.y, 2)+ powf(sample.z, 2));
}