#include "includes.h"
__global__ void k_Simple( const float* p_Input, float* p_Output, int p_Width, int p_Height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
p_Output[index] = p_Input[index];
p_Output[index + 1] = p_Input[index + 1];
p_Output[index + 2] = p_Input[index + 2];
p_Output[index + 3] = p_Input[index + 3];
}}