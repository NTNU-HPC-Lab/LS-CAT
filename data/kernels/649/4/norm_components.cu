#include "includes.h"
__global__ void norm_components(float* N, int npix, float* norm) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < npix) {
norm[i] = fmaxf(1e-10, sqrtf(N[i] * N[i] + N[npix + i] * N[npix + i] + N[npix * 2 + i] * N[npix * 2 + i]));
}
}