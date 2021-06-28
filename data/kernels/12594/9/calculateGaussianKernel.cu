#include "includes.h"
__global__ void calculateGaussianKernel(float *gaussKernel, const float sigma, int halfKernelWidth){

/// pixel index of this thread
/// this makes the normal curve
int i = threadIdx.x - halfKernelWidth;
extern __shared__ float s_gaussKernel[];
__shared__ float sum;

/// this kernel must allocate 'kernelWidth' threads
s_gaussKernel[threadIdx.x] = (__fdividef(1,(sqrtf(2*M_PI*sigma))))*expf((-1)*(__fdividef((i*i),(2*sigma*sigma))));

__syncthreads();

/// Thread 0 sum all the gassian kernel array
// This is not so bad because the array is always short
if (!threadIdx.x) {
int th;
sum = 0;
for(th = 0; th<blockDim.x; th++) sum += s_gaussKernel[th];
}

__syncthreads();

gaussKernel[threadIdx.x] = s_gaussKernel[threadIdx.x]/sum;

}