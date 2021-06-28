#include "includes.h"
__global__ void updateState(float *B, float *external, int dim, float timestep, float noise, int length, int totalIterations, int iterationNum, float L, float M) {
int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
if (index >= length && index < length + dim) {
int neuronNum = index % dim;
float input = B[index] + external[neuronNum * (totalIterations) + iterationNum];
float old_output = B[index - dim];
float d_layers = (-1 * old_output) + 1 / (1 + expf(-1 * L * (input - M)));

// create random number generator
curandState_t state;
curand_init (blockIdx.x * 1000 + threadIdx.x + clock64(), 0, 0, &state);
float random = curand_normal(&state);
float guassian_noise = noise * random * sqrt(timestep);
B[index] = old_output + d_layers * timestep + guassian_noise;
}
}