#include "includes.h"
__global__ void normalize_N(float* N, float* norm, int npix_per_component) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int c = blockIdx.y*blockDim.y + threadIdx.y;
if (i < npix_per_component) {
N[c*npix_per_component + i] = N[c*npix_per_component + i] / norm[i];
}
}