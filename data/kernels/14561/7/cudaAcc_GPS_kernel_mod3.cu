#include "includes.h"
__global__ void cudaAcc_GPS_kernel_mod3( int NumDataPoints, float2* FreqData, float* PowerSpectrum)
{
const int sidx = (blockIdx.x * blockDim.x + threadIdx.x);

float ax,ay;

if ( sidx < NumDataPoints )
{
ax = FreqData[sidx].x;
ay = FreqData[sidx].y;
PowerSpectrum[sidx] =  __fadd_rn( __fmul_rn(ax,ax),__fmul_rn(ay,ay));
}
}