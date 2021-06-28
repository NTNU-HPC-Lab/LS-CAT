#include "includes.h"
__global__ void unpackSourceKernelGrayscale16(uint16_t* dst, unsigned pitch, const cudaSurfaceObject_t src, unsigned width, unsigned height) {
const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) {
float val;
surf2Dread(&val, src, x * sizeof(float), y);
const float inMilliMeters = val * 1000.f;
const uint16_t u16 = (uint16_t)max(0.f, min((float)USHRT_MAX, round(inMilliMeters)));
dst[y * pitch + x] = u16;
}
}