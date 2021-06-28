#include "includes.h"
__global__ void increment(char* data, size_t length)
{
size_t global_index = threadIdx.x + blockIdx.x * blockDim.x;
if (global_index < length)
data[global_index]++;
}