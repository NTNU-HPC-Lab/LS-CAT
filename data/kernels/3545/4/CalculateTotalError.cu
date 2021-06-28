#include "includes.h"
// Device code for ICP computation
// Currently working only on performing rotation and translation using cuda


#ifndef _ICP_KERNEL_H_
#define _ICP_KERNEL_H_



#define TILE_WIDTH 256




















#endif // #ifndef _ICP_KERNEL_H_
__global__ void CalculateTotalError(double * distance_d, int size_data)
{
__shared__ double error_s[2*TILE_WIDTH];

unsigned int t = threadIdx.x;
unsigned int start = 2*blockDim.x*blockIdx.x;

if(start + t < size_data)
error_s[t] = distance_d[start + t];
else
error_s[t] = 0.0f;
if(start + blockDim.x + t < size_data)
error_s[blockDim.x + t] = distance_d[start + blockDim.x + t];
else
error_s[blockDim.x + t] = 0.0f;

for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1)
{
__syncthreads();
if(t < stride)
error_s[t] += error_s[t + stride];
}

if(t == 0)
distance_d[blockIdx.x] = error_s[t];

}