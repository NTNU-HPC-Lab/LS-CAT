#include "includes.h"
__global__ void A_for_lightning_estimation(float* rho, float* N, int npix, float* A) {
int i = blockIdx.x*blockDim.x + threadIdx.x; // pixel index
int c = blockIdx.y*blockDim.y + threadIdx.y; // channel index
int h = blockIdx.z*blockDim.z + threadIdx.z; // harmonic index
if (i < npix) {
A[c*npix * 4 + h*npix + i] = rho[c*npix + i] * N[h*npix + i];
}
}