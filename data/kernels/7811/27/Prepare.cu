#include "includes.h"
__global__ void Prepare(float* p_Input, float* p_Output, int p_Width, int p_Height, int p_Display) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
float ramp = (float)x / (float)(p_Width - 1);
p_Output[index] = p_Display == 1 ? ramp : p_Input[index];
p_Output[index + 1] = p_Display == 1 ? ramp : p_Input[index + 1];
p_Output[index + 2] = p_Display == 1 ? ramp : p_Input[index + 2];
p_Output[index + 3] = 1.0f;
if (p_Display == 2) {
p_Input[index] = ramp;
p_Input[index + 1] = ramp;
p_Input[index + 2] = ramp;
}}}