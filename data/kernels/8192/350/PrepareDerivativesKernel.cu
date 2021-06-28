#include "includes.h"
__global__ void PrepareDerivativesKernel(float* input, float* lastInput, float* derivatives, int inputWidth, int inputHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;
int size =  inputWidth * inputHeight;

if (id < size)
{
float mul = 100000;
//I_x, I_y
float I_x = mul * derivatives[id];
float I_y = mul * derivatives[size + id];

//I_t
float input_dt = mul * (input[id] - lastInput[id]);
lastInput[id] = input[id];

// I_x * I_y
derivatives[2 * size + id] = I_x * I_y;
// I_x * I_t
derivatives[3 * size + id] = I_x * input_dt;
// I_x * I_t
derivatives[4 * size + id] = I_y * input_dt;
// I_x ^ 2
derivatives[id] = I_x * I_x;
// I_y ^ 2
derivatives[size + id] = I_y * I_y;
}
}