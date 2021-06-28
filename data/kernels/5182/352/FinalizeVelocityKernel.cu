#include "includes.h"
__global__ void FinalizeVelocityKernel(float* velocities, float* globalFlow, int inputWidth, int inputHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;
int size =  inputWidth * inputHeight;

if (id < size)
{
float globalFlowL = sqrtf(globalFlow[0] * globalFlow[0] + globalFlow[1] * globalFlow[1]);
float velocityL = sqrtf(velocities[id] * velocities[id]  + velocities[size + id] * velocities[size + id]);

if (globalFlowL > 0 && velocityL > 0) {

float dot = (globalFlow[0] * velocities[id] + globalFlow[1] * velocities[size + id]) / (globalFlowL * velocityL);

if (dot > 0.7) {
velocities[id] = 0;
velocities[size + id] = 0;
}
}
}
}