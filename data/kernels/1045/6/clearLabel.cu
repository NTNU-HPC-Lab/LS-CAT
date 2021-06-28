#include "includes.h"
__global__ void clearLabel(float *prA, float *prB, unsigned int num_nodes, float base)
{
unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
if(id < num_nodes)
{
prA[id] = base + prA[id] * 0.85;
prB[id] = 0;
}
}