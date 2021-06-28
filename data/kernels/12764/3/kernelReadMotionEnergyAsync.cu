#include "includes.h"
__global__ void kernelReadMotionEnergyAsync(float* gpuConvBufferl1, float* gpuConvBufferl2, int ringBufferIdx, int bsx, int bsy, int n, float* gpuEnergyBuffer)
{
int bufferPos = threadIdx.x + blockIdx.x * blockDim.x;
if(bufferPos < n) {
// Offset in ringbuffer
int bufferPosConv = bufferPos + ringBufferIdx*bsx*bsy;
// Get answer from two corresponding buffers and compute motion energy
float l1 = gpuConvBufferl1[bufferPosConv];
float l2 = gpuConvBufferl2[bufferPosConv];

// Compute motion energy
gpuEnergyBuffer[bufferPos] = sqrt(l1*l1+l2*l2);
}
}