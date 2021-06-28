#include "includes.h"
__global__ void unpackSourceKernelF32C1(float* dst, unsigned pitch, const cudaSurfaceObject_t src, unsigned width, unsigned height) {
const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) {
// yeah, we could use a memcpy
float val;
surf2Dread(&val, src, x * sizeof(float), y);
dst[y * pitch + x] = val;
}
}