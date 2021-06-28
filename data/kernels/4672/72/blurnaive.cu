#include "includes.h"
__global__ void blurnaive(float* matrix, float* output, int firstFrame, int numFrames, int frameCount, int max, int length){
// int frame = firstFrame + (blockIdx.y*blockDim.y+ threadIdx.y);
int chan = (blockIdx.x*blockDim.x+ threadIdx.x)<<1;
float amp = 0.0f;
float freq = 0.0f;
int frame;
if (chan < length) {
for (frame = firstFrame; frame != frameCount; frame = (frame + 1) % max) {
amp += matrix[frame*length+chan];
freq += matrix[frame*length+chan+1];
}
output[chan] = (float) (amp / numFrames);
output[chan+1] = (float) (freq / numFrames);
}
}