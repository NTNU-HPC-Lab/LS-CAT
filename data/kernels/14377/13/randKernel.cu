#include "includes.h"
__global__ void randKernel(float* out, curandState* states, float min, float scale) {
int id  = blockIdx.x * blockDim.x + threadIdx.x;
out[id] = curand_uniform(&states[id]) * scale + min;
}