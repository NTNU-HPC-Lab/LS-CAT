#include "includes.h"
__global__ void compute_shared(const int* destination_offsets, const int* source_indices, const int* out_degrees, const int node_count, const float* input, float *output)
{
int dest = blockDim.x*blockIdx.x + threadIdx.x;
__shared__ int s_dest_off[BLOCK_SIZE + 1];
if (dest<node_count)
{
s_dest_off[threadIdx.x] = destination_offsets[dest];
if (threadIdx.x == BLOCK_SIZE - 1 || dest == node_count - 1)
{
s_dest_off[threadIdx.x + 1] = destination_offsets[dest + 1];
}
__syncthreads();
int srcStart = s_dest_off[threadIdx.x];
int srcEnd = s_dest_off[threadIdx.x + 1];
int in_degree = srcEnd - srcStart;
float rank = 0;
if (in_degree>0)
{
for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
{
int src = source_indices[srcIdx];
float contrib = ((input[src] * DECAY) / out_degrees[src]);
rank = rank + contrib;
}
}
output[dest] = rank + (1 - DECAY);
}
}