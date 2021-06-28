#include "includes.h"
__global__ void generateCurve(float t, float dx, float* out, const float ZMIN, const size_t ZSIZE) {
int offset = blockIdx.x * blockDim.x + threadIdx.x;

float z = ZMIN + offset * dx;
if (offset < ZSIZE) {
out[3 * offset]     = cos(z * t + t) / z;
out[3 * offset + 1] = sin(z * t + t) / z;
out[3 * offset + 2] = z + 0.1 * sin(t);
}
}