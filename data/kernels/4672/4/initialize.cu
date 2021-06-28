#include "includes.h"
__global__ void initialize(float* matrix, float sr, int numFrames, int length) {
int frame = blockIdx.y*blockDim.y + threadIdx.y;
int chan = (blockIdx.x*blockDim.x + threadIdx.x) << 1;
if ((frame < numFrames) && (chan < length)) {
matrix[frame*length+chan] = 0.0f;
matrix[frame*length+chan+1] = chan * sr / (length-2);
}
}