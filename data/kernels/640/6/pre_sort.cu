#include "includes.h"




__global__ void pre_sort(unsigned int *in, unsigned int *in_pos, unsigned int *out, unsigned int *out_pos, unsigned int n, unsigned int nBins, unsigned int mask, unsigned int current_bits, unsigned int *d_hist)
{
extern __shared__ unsigned int pre_sort_blk_data[];
unsigned int* blk_value = pre_sort_blk_data;
unsigned int* blk_pos = pre_sort_blk_data + blockDim.x;
unsigned int* blk_hist = pre_sort_blk_data + 2*blockDim.x;
unsigned int* blk_Scan = pre_sort_blk_data + nBins + 2*blockDim.x;

int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n)
{
blk_value[threadIdx.x] = in[i];
blk_pos[threadIdx.x] = in_pos[i];
}
__syncthreads();

//Hist
for(int j = threadIdx.x; j < nBins; j += blockDim.x)
{
blk_hist[j] = 0;
blk_Scan[j] = 0;
}
__syncthreads();

unsigned int bin = (blk_value[threadIdx.x] >> current_bits) & mask;
atomicAdd(&blk_hist[bin], 1);
atomicAdd(&blk_Scan[bin], 1);
__syncthreads();

//Scan
for (int stride = 1; stride < nBins; stride *= 2)
{
for (int k = threadIdx.x; k < nBins; k += blockDim.x)
{
int inVal;
if (k >= stride)
inVal = blk_Scan[k - stride];
__syncthreads();
if (k >= stride)
blk_Scan[k] += inVal;
__syncthreads();
}
}
__syncthreads();

for (int i = threadIdx.x; i < nBins; i += blockDim.x)
blk_Scan[i] -= blk_hist[i];
__syncthreads();

//Scatter
if (threadIdx.x == 0)
{
for (int i = 0; i < blockDim.x; i++)
{
unsigned int bin = (blk_value[i] >> current_bits) & mask;
out[blk_Scan[bin] + blockIdx.x*blockDim.x] = blk_value[i];
out_pos[blk_Scan[bin] + blockIdx.x*blockDim.x] = blk_pos[i];
blk_Scan[bin]++;
}
}
}