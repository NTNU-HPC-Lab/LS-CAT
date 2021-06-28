#include "includes.h"
__global__ void MaskInput( float* image, float* mask, float* maskedValues, float* output, int count ) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;

if (id < count) {
output[id] = image[id] * mask[id] + maskedValues[id] * (1.0f - mask[id]);
}
}