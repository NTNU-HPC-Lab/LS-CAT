#include "includes.h"
__global__ void calcCDF(float *cdf, unsigned int *histo, int imageWidth, int imageHeight, int length) {

__shared__ float partialScan[SIZE_CDF];
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < SIZE_CDF && i < 256) {
partialScan[i] = (float) histo[i] / (float) (imageWidth * imageHeight);

}
__syncthreads();

for (unsigned int stride = 1; stride <= SIZE_HISTO; stride *= 2) {
unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
if (index < SIZE_CDF && index < length)
partialScan[index] += partialScan[index - stride];
__syncthreads();
}

for (unsigned int stride = SIZE_HISTO / 2; stride > 0; stride /= 2) {
__syncthreads();
unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
if (index + stride < SIZE_CDF && index + stride < length) {
partialScan[index + stride] += partialScan[index];
}
}

__syncthreads();
if (i < SIZE_CDF && i < 256) {
cdf[i] += partialScan[threadIdx.x];
}
}