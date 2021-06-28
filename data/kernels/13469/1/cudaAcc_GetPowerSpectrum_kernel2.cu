#include "includes.h"
#define B 2


/*
*/

__global__ void cudaAcc_GetPowerSpectrum_kernel2( int NumDataPoints, float2* FreqData, float* PowerSpectrum)
{
const int i = blockIdx.x * blockDim.x*B + threadIdx.x;

float ax[B];
float ay[B];

#pragma unroll
for (int k=0;k<B;k++)
{
ax[k] = FreqData[i+k*blockDim.x].x;
ay[k] = FreqData[i+k*blockDim.x].y;
}
//		PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;

#pragma unroll
for (int k=0;k<B;k++)
{
PowerSpectrum[i+k*blockDim.x] = __fadd_rn( __fmul_rn(ax[k],ax[k]),__fmul_rn(ay[k],ay[k]));
}
}