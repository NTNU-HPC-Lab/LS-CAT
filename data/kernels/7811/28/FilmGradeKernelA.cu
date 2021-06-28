#include "includes.h"
__global__ void FilmGradeKernelA( float* p_Input, int p_Width, int p_Height, float p_Exp) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if(x < p_Width && y < p_Height) {
const int index = (y * p_Width + x) * 4;
p_Input[index] = p_Input[index] + p_Exp * 0.01f;
}}