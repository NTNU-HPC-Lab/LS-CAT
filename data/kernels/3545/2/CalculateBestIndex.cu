#include "includes.h"
// Device code for ICP computation
// Currently working only on performing rotation and translation using cuda


#ifndef _ICP_KERNEL_H_
#define _ICP_KERNEL_H_



#define TILE_WIDTH 256




















#endif // #ifndef _ICP_KERNEL_H_
__global__ void CalculateBestIndex(double * distance_d, int * bin_index_d, int size_data)
{
__shared__ double distance_s[2*TILE_WIDTH];
__shared__ unsigned int bin_smallest_index[2*TILE_WIDTH];
unsigned int t = threadIdx.x;
unsigned int start = 2*blockDim.x*blockIdx.x;

if(start + t < size_data)
{
distance_s[t] = distance_d[start + t];
bin_smallest_index[t] = bin_index_d[start + t];
}
else
{
distance_s[t] = 65535;
bin_smallest_index[t] = 0;
}
if(start + blockDim.x + t < size_data)
{
distance_s[blockDim.x + t] = distance_d[start + blockDim.x + t];
bin_smallest_index[blockDim.x + t] = bin_index_d[start + blockDim.x + t];
}
else
{
distance_s[blockDim.x + t] = 65535;
bin_smallest_index[blockDim.x + t] = 0;
}

for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1)
{
__syncthreads();
if(t < stride)
if(distance_s[t] > distance_s[stride + t])
{
bin_smallest_index[t] = bin_smallest_index[stride + t];
distance_s[t] = distance_s[stride + t];
}

}

if(t == 0)
{
distance_d[blockIdx.x] = distance_s[t];
bin_index_d[blockIdx.x] = bin_smallest_index[t];
}


}