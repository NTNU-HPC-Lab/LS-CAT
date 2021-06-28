#include "includes.h"
__global__ void LoadAddVecSecond(float *vector , float2 *FFT) {

int idx = threadIdx.x + blockIdx.x*blockDim.x; // this should span the full range of the vector
FFT[idx].x *= vector[idx]/sqV;
FFT[idx].y *= vector[idx]/sqV;
}