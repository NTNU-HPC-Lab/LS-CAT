#include "includes.h"
__global__ void updateState(float *B, float *external, int dim, float timestep, int length, float L, float M) {
int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
if (index < length + dim) {
float input = B[index] + external[index];
float old_output = B[index - dim];
float d_layers = (-1 * old_output) + 1 / (1 + expf(-1 * L * (input - M)));
B[index] = old_output + d_layers * timestep;
}
}