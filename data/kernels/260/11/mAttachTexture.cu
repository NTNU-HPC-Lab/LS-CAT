#include "includes.h"
__global__ void mAttachTexture(uint8_t *frame, float *dense) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
frame[Idx] = (dense[Idx] > 255.0)? 255:(uint8_t)(dense[Idx]);
}