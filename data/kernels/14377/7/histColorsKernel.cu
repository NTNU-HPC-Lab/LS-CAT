#include "includes.h"
__global__ void histColorsKernel(float* histColors, curandState* states) {
int bin = blockIdx.x * blockDim.x + threadIdx.x;

histColors[3 * bin + 0] = curand_uniform(&states[bin]);
histColors[3 * bin + 1] = curand_uniform(&states[bin]);
histColors[3 * bin + 2] = curand_uniform(&states[bin]);
}