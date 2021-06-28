#include "includes.h"
__global__ void PIDControllerKernel(float* input, float* goal, float* output, float* previousError, float* integral)
{
int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (id < D_COUNT)
{
float error = input[id] - goal[id];
integral[id] = D_INTEGRAL_DECAY * integral[id] + error;
float derivative = error - previousError[id];

previousError[id] = error;


float out = D_OFFSET + D_PROPORTIONAL_GAIN * error + D_INTEGRAL_GAIN * integral[id] + D_DERIVATIVE_GAIN * derivative;
if (out > D_MAX_OUTPUT)
out = D_MAX_OUTPUT;
if (out < D_MIN_OUTPUT)
out = D_MIN_OUTPUT;

output[id] = out;
}
}