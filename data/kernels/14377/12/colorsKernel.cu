#include "includes.h"
__global__ void colorsKernel(float* colors, curandState* states) {
int id = blockIdx.x * blockDim.x + threadIdx.x;

colors[3 * id + 0] = curand_uniform(&states[id]);
colors[3 * id + 1] = curand_uniform(&states[id]);
colors[3 * id + 2] = curand_uniform(&states[id]);
}