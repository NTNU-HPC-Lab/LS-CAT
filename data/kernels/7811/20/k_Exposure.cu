#include "includes.h"
__global__ void k_Exposure( float* p_Input, int p_Width, int p_Height, float p_Exposure) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
p_Input[index] = p_Input[index] * exp2(p_Exposure);
p_Input[index + 1] = p_Input[index + 1] * exp2(p_Exposure);
p_Input[index + 2] = p_Input[index + 2] * exp2(p_Exposure);
}}