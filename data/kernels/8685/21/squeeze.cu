#include "includes.h"
__global__ void squeeze(float *B, int dim, int length, float L, float M) {
int index = (blockIdx.x * blockDim.x) + threadIdx.x + length;
if (index < length + dim) {
B[index] = 1 / (1 + expf(-1 * L * (B[index] - M)));
}
}