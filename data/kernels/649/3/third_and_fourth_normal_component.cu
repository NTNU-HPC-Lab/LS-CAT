#include "includes.h"
__global__ void third_and_fourth_normal_component(float* z, float* xx, float* yy, float* zx, float* zy, int npix, float* N3) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < npix) {
N3[i] = -z[i] - (xx[i]) * zx[i] - (yy[i]) * zy[i];
N3[npix + i] = 1;
}
}