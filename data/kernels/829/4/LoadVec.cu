#include "includes.h"
__global__ void LoadVec(float *vector , float2 *FFT) {
int idx = threadIdx.x + blockIdx.x*blockDim.x; // this should span the full range of the vector
FFT[idx].x = vector[idx]; // The real part is replaced by the vector value
FFT[idx].y = 0.0f;        // The imaginary part is zero. The following kernel also replaces the imaginary part
}