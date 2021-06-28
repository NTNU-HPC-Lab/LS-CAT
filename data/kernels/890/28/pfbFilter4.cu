#include "includes.h"
__global__ void pfbFilter4(float *filtered, float *unfiltered, float *taps, const int ntaps) {

const int nfft = blockDim.x;
const int i = threadIdx.x + threadIdx.y*blockDim.x*4 + blockIdx.x*blockDim.x*blockDim.y*4;

filtered[i] = unfiltered[i] * taps[threadIdx.x];
filtered[i+nfft] = unfiltered[i+nfft] * taps[threadIdx.x];
filtered[i+nfft*2] = unfiltered[i+nfft*2] * taps[threadIdx.x];
filtered[i+nfft*3] = unfiltered[i+nfft*3] * taps[threadIdx.x];
for (int j=1; j<ntaps; j++) {
filtered[i] += unfiltered[i + j*nfft] * taps[threadIdx.x + j*nfft];
filtered[i+nfft] += unfiltered[i + (j+1)*nfft] * taps[threadIdx.x + j*nfft];
filtered[i+nfft] += unfiltered[i + (j+2)*nfft] * taps[threadIdx.x + j*nfft];
filtered[i+nfft] += unfiltered[i + (j+3)*nfft] * taps[threadIdx.x + j*nfft];
}
}