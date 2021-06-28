#include "includes.h"
__global__ void GPU_vector_add(int* left, int* right, int* result)
{
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
result[idx] = left[idx] + right[idx];
}