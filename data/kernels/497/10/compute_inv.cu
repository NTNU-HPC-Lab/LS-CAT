#include "includes.h"
__global__ void compute_inv(const int* destination_offsets, const int* source_indices, const float* out_degrees, const int node_count, const float* input, float *output)
{
int dest = blockDim.x*blockIdx.x + threadIdx.x;
if (dest<node_count)
{
int srcStart = destination_offsets[dest];
int srcEnd = destination_offsets[dest + 1];
int in_degree = srcEnd - srcStart;
float rank = 0;
if (in_degree>0)
{
for (int srcIdx = srcStart; srcIdx<srcEnd; ++srcIdx)
{
int src = source_indices[srcIdx];
float contrib = ((input[src] * DECAY) * out_degrees[src]);
rank = rank + contrib;
}
}
output[dest] = rank + (1 - DECAY);
}
}