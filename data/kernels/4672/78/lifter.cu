#include "includes.h"
__global__ void lifter(float* cepstrum, int nCoefs, int nhalf) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
int k = i + nCoefs;
if (k < nhalf+2-nCoefs) {
cepstrum[k] = 0.0;   // kill all the cepstrum coefficients above nCoefs
}
}