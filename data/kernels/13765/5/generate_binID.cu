#include "includes.h"
__global__ void generate_binID(const float* dIn, int* out, const int binNumber, const float lumMin, const float lumMax, const int size) {
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i>size)
{
return;
}

float range = lumMax - lumMin;
int bin = ((dIn[i] - lumMin) / range) * binNumber;

out[i] = bin;
}