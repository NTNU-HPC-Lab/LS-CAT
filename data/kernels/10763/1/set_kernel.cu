#include "includes.h"
__global__ void set_kernel(REAL* dst, REAL const value, std::size_t const count)
{
std::size_t const index = blockIdx.x * blockDim.x + threadIdx.x;

if (index >= count)
return;

dst[index] = value;
}