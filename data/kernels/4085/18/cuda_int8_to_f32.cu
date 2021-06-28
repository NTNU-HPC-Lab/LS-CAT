#include "includes.h"
__global__ void cuda_int8_to_f32(int8_t* input_int8, size_t size, float *output_f32, float multipler)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) output_f32[idx] = input_int8[idx] * multipler; // 7-bit (1-bit sign)

}