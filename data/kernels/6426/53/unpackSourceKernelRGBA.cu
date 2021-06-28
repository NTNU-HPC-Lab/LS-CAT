#include "includes.h"
__global__ void unpackSourceKernelRGBA(uint32_t* dst, unsigned pitch, const cudaSurfaceObject_t src, unsigned width, unsigned height) {
const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) {
// yeah, we could use a memcpy
uint32_t val;
surf2Dread(&val, src, x * sizeof(uint32_t), y);
dst[y * pitch + x] = val;
}
}