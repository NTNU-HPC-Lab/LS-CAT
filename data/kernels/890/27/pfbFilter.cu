#include "includes.h"
__global__ void pfbFilter(float *filtered, float *unfiltered, float *taps, const int ntaps) {

const int nfft = blockDim.x;
const int i = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x * blockDim.x * blockDim.y;

filtered[i] = unfiltered[i] * taps[threadIdx.x];
for (int j=1; j<ntaps; j++) {
filtered[i] += unfiltered[i + j*nfft] * taps[threadIdx.x + j*nfft];
}
}