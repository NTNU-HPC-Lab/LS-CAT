#include "includes.h"
__global__ void IntervalToBinaryVector(float input, float* outputs, int steps)
{
int id = blockDim.x*blockIdx.y*gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

if(id < steps)
{
float fraction = 1.0f / steps;
outputs[id] = input >= fraction * id && input <= fraction * (id + 1);
}
}