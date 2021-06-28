#include "includes.h"
__global__ void cuda_multiply_f32(float *input_output, size_t size, float multipler)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) input_output[idx] = input_output[idx] * multipler; // 7-bit (1-bit sign)

}