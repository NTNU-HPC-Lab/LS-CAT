#include "includes.h"
__global__ void thresholdSIMD(unsigned int *data, unsigned int threshold)
{
int thread = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
unsigned int *ptr = data + thread;

*ptr = __vcmpgeu4(*ptr, threshold);
}