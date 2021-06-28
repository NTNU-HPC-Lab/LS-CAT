#include "includes.h"
__global__ void uniformAdd(unsigned int n, unsigned int *data, unsigned int *inter)
{

__shared__ unsigned int uni;
if (threadIdx.x == 0) { uni = inter[blockIdx.x]; }
__syncthreads();

unsigned int g_ai = blockIdx.x*2*blockDim.x + threadIdx.x;
unsigned int g_bi = g_ai + blockDim.x;

if (g_ai < n) { data[g_ai] += uni; }
if (g_bi < n) { data[g_bi] += uni; }
}