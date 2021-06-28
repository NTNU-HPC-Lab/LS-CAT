#include "includes.h"
__global__ void pointGenKernel(float* points, float* dirs, int nBBS0, int nelems, float minimum, float step) {
int k = blockIdx.x / nBBS0;
int i = blockDim.x * (blockIdx.x - k * nBBS0) + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;

if (i < nelems && j < nelems && k < nelems) {
float x = minimum + i * step;
float y = minimum + j * step;
float z = minimum + k * step;

int id = i + j * nelems + k * nelems * nelems;

points[3 * id + 0] = x;
points[3 * id + 1] = y;
points[3 * id + 2] = z;

dirs[3 * id + 0] = x - 10.f;
dirs[3 * id + 1] = y - 10.f;
dirs[3 * id + 2] = z - 10.f;
}
}