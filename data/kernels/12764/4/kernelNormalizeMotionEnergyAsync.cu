#include "includes.h"
__global__ void kernelNormalizeMotionEnergyAsync(int bsx, int bsy, int n, float alphaPNorm, float alphaQNorm, float betaNorm, float sigmaNorm, float* gpuEnergyBuffer)
{
int bufferPos = threadIdx.x + blockIdx.x * blockDim.x;
float sigmaNorm2_2 = 2*sigmaNorm*sigmaNorm;
if(bufferPos < n) {
int bx,by;
int bxy = bufferPos / (bsx*bsy);
bx = bxy % bsx;
by = bxy / bsx;
// Read energy
float I = gpuEnergyBuffer[bufferPos];
float q_i = 0;
// Normalize over 5x5 region
for(int y = -2; y <= 2; y++) {
int by_ = by + y;

if(by_ < 0 || by_ >= bsy)
continue;

for(int x = -2; x <= 2; x++) {
int bx_ = bx + x;

if(bx_ < 0 || bx_ >= bsx ||
(bx == bx_ && by == by_))
continue;
// TODO
// Each thread computes the same
float gaus = 1/(sigmaNorm2_2*M_PI)* exp(-(bx_*bx_ + by_*by_)/sigmaNorm2_2);
// TODO Use shared memory to avoid extra global memory access
q_i += gpuEnergyBuffer[by_*bsx+bx_]*gaus;
}
}
q_i /= alphaQNorm;

// Compute p_i
float p_i = (I*betaNorm)/(alphaPNorm + I + q_i);

// Use normalized value
gpuEnergyBuffer[bufferPos] = p_i;
}
}