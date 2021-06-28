#include "includes.h"
__global__ void idx_print()
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int warp_idx = threadIdx.x / warpSize;
int lane_idx = threadIdx.x & (warpSize - 1);

if ((lane_idx & (warpSize/2 - 1)) == 0)
//  thread, block, warp, lane"
printf(" %5d\t%5d\t %2d\t%2d\n", idx, blockIdx.x, warp_idx, lane_idx);
}