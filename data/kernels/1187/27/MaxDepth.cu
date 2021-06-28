#include "includes.h"
__global__ void MaxDepth (const int n, const float *bottom_data, const int step, const int depth, float *idx){

int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= n)
{
return;
}
int base = index / step * step * depth + index % step;
int k = 0;
for (int i = 1; i < depth; i++)
if (bottom_data[base + k * step] < bottom_data[base + i * step])
k = i;
idx[index] = k;
}