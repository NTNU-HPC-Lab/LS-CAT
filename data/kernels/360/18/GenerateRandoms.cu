#include "includes.h"
__global__ void GenerateRandoms(int size, int iterations, unsigned int *randoms, unsigned int *seeds) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int z = seeds[idx];
int offset = idx;
int step = 32768;

for (int i = 0; i < iterations; i++)
{
if (offset < size)
{
unsigned int b = (((z << 13) ^ z) >> 19);
z = (((z & UINT_MAX) << 12) ^ b);
randoms[offset] = z;
offset += step;
}
}
}