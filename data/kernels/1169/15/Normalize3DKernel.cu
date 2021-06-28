#include "includes.h"
__global__ void Normalize3DKernel ( const unsigned short *d_src, const float *d_erosion, const float *d_dilation, float *d_dst, float min_intensity, const int width, const int height, const int depth ) {
const int baseX = blockIdx.x * blockDim.x + threadIdx.x;
const int baseY = blockIdx.y * blockDim.y + threadIdx.y;
const int baseZ = blockIdx.z * blockDim.z + threadIdx.z;

const int idx = (baseZ * height + baseY) * width + baseX;
const float intensity = (float)d_src[idx];
d_dst[idx] = (intensity >= min_intensity) ? (intensity-d_erosion[idx]) / (d_dilation[idx] - d_erosion[idx]) : 0;
}