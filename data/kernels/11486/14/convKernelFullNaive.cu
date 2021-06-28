#include "includes.h"
__global__ void convKernelFullNaive(float* d_Input, float* d_Output, float* d_Kernel, int  imageW, int imageH, int kernelR)
{
int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;

int loc = row * imageW + col;

float s = 0;
float t = 0;

for (int i = -KERNAL_RAD; i <= KERNAL_RAD; i++)
for (int j = -KERNAL_RAD; j <= KERNAL_RAD; j++)
{
t = 0;

if (row  + i >= 0 && row  + i < imageH && col  + j >= 0 && col  + j < imageW )
t = d_Input[loc + i * imageW + j];

s += t * d_Kernel[(KERNAL_RAD - i) * (KERNAL_RAD + KERNAL_RAD + 1) + KERNAL_RAD - j];
}
d_Output[loc] = s;
}