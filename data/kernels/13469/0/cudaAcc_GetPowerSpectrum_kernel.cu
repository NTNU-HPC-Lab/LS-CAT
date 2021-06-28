#include "includes.h"
#define B 2


/*
*/

__global__ void cudaAcc_GetPowerSpectrum_kernel( int NumDataPoints, float2* FreqData, float* PowerSpectrum) {
const int i = blockIdx.x * blockDim.x + threadIdx.x;

//	if (i < NumDataPoints) {
float ax = FreqData[i].x;
float ay = FreqData[i].y;
//		PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;
PowerSpectrum[i] = __fadd_rn( __fmul_rn(ax,ax),__fmul_rn(ay,ay));
//	}
}