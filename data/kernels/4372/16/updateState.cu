#include "includes.h"
__global__ void updateState(double *B, double *external, double *lamBeta, int dim, float timestep, double noise, int length, int totalIterations, int iterationNum) {
int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
if (index >= length && index < length + dim) {
int neuronNum = index % dim;
double lam = lamBeta[neuronNum * 2];
double beta = lamBeta[neuronNum * 2 + 1];

double input = B[index] + external[neuronNum * (totalIterations) + iterationNum];
double old_output = B[index - dim];
double d_layers = (-1 * old_output) + 1 / (1 + expf(-1 * lam * (input - beta)));

// create random number generator
curandState_t state;
curand_init (blockIdx.x * 1000 + threadIdx.x + clock64(), 0, 0, &state);
float random = curand_normal(&state);
double guassian_noise = noise * random * sqrt(timestep);
B[index] = old_output + d_layers * timestep + guassian_noise;
}
}