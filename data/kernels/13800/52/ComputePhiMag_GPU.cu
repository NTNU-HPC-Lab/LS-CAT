#include "includes.h"
__global__ void ComputePhiMag_GPU(float* phiR, float* phiI, float* phiMag, int numK) {
int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
if (indexK < numK) {
float real = phiR[indexK];
float imag = phiI[indexK];
phiMag[indexK] = real*real + imag*imag;
}
}