#include "includes.h"


// Bodies_input array contains position [0,1], velocity [2,3], mass [4]
// Bodies_output array contains position [0,1], velocity [2,3], mass [4]; mass is not used here

__global__ void forces_and_step(double *bodies_input, double *bodies_output, unsigned int count, double dt, uint16_t bods_per_thread)
{
unsigned int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * 5 * bods_per_thread;
//unsigned int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * 5;

for (int b = 0; b < bods_per_thread; b++)
{
// If there are more threads than data discard the extra computations to stay in defined memory
if (index < count * 5)
{
// Calculate force for this particle
double fx = 0.0;
double fy = 0.0;
for (uint32_t i = 0; i < count * 5; i += 5)
{
double dir_x = bodies_input[i] - bodies_input[index];
double dir_y = bodies_input[i + 1] - bodies_input[index + 1];
// Make sure there is no division by zero
if (dir_x == 0.0 && dir_y == 0.0)
continue;
fx += G_CONSTANT * bodies_input[i + 4] * bodies_input[index + 4] * dir_x
/ pow(sqrt(dir_x * dir_x + dir_y * dir_y), 3.0);
fy += G_CONSTANT * bodies_input[i + 4] * bodies_input[index + 4] * dir_y
/ pow(sqrt(dir_x * dir_x + dir_y * dir_y), 3.0);
}

// Integration
bodies_output[index + 2] = bodies_input[index + 2] + (fx / bodies_input[index + 4]) * dt;
bodies_output[index + 3] = bodies_input[index + 3] + (fy / bodies_input[index + 4]) * dt;
bodies_output[index] = bodies_input[index] + bodies_output[index + 2] * dt;
bodies_output[index + 1] = bodies_input[index + 1] + bodies_output[index + 3] * dt;
bodies_output[index + 4] = bodies_input[index + 4];
}
index += 5;
}
}