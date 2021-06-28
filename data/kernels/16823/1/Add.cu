#include "includes.h"
__global__ void Add(float* d_a, float* d_b, float* d_c, int N)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id < N)

d_c[id] = d_a[id] + d_b[id];

}