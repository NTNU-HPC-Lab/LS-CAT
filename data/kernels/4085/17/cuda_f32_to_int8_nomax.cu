#include "includes.h"
__global__ void cuda_f32_to_int8_nomax(float* input_f32, size_t size, int8_t *output_int8, float multipler)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) output_int8[idx] = input_f32[idx] * multipler; // 7-bit (1-bit sign)

}