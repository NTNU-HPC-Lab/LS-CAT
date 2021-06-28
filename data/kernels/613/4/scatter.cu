#include "includes.h"
//Udacity HW 4
//Radix Sorting





__global__ void scatter(unsigned int *in,unsigned int *in_pos, unsigned int *out, unsigned int *out_pos, unsigned int n, unsigned int *d_histScan, unsigned int mask, unsigned int current_bits, unsigned int nBins)
{
if (threadIdx.x == 0)
{
unsigned int start = blockIdx.x*blockDim.x;
for (int i = start; i < min(n, start + blockDim.x) ; i++)
{
unsigned int bin = (in[i] >> current_bits) & mask;
out[d_histScan[blockIdx.x + bin*gridDim.x]] = in[i];
out_pos[d_histScan[blockIdx.x + bin*gridDim.x]] = in_pos[i];
d_histScan[blockIdx.x + bin*gridDim.x]++;
}
}
}