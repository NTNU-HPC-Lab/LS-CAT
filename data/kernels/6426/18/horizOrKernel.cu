#include "includes.h"
__global__ void horizOrKernel(const uint32_t* __restrict__ contrib, uint32_t* __restrict__ rowHasImage, unsigned panoWidth, unsigned panoHeight) {
unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
const uint32_t* rowp = contrib + panoWidth * row;

if (row < panoHeight) {
uint32_t accum = 0;
for (unsigned col = 0; col < panoWidth; ++col) {
accum |= rowp[col];
}
rowHasImage[row] = accum;
}
}