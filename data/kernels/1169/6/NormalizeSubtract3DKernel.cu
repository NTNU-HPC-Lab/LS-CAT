#include "includes.h"
__global__ void NormalizeSubtract3DKernel(float * img_src, const float * img_sub, const int width, const int height, const int depth, float normalizer) {
const int baseX = blockIdx.x * SUB_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * SUB_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z * SUB_BLOCKDIM_Z + threadIdx.z;

const int idx = (baseZ * height + baseY) * width + baseX;
img_src[idx] = (img_src[idx] - img_sub[idx]) * normalizer;

}