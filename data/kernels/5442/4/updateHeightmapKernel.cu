#include "includes.h"
///////////////////////////////////////////////////////////////////////////////

//Round a / b to nearest higher integer value
__global__ void updateHeightmapKernel(float*  heightMap, float2* ht, unsigned int width){
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
unsigned int i = y*width+x;

float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;
heightMap[i] = ht[i].x * sign_correction;
}