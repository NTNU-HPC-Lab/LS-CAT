#include "includes.h"
__global__ void pfbFilterShared(float *filtered, float *unfiltered, float *taps, const int ntaps) {
extern __shared__ float shared_taps[];

const int nfft = blockDim.x;
const int i = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
if (i<ntaps*nfft) {
shared_taps[i] = taps[i];
}
__syncthreads();


filtered[i] = unfiltered[i] * shared_taps[threadIdx.x];
for (int j=1; j<ntaps; j++) {
filtered[i] += unfiltered[i + j*nfft] * shared_taps[threadIdx.x + j*nfft];
}
}