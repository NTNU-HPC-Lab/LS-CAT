#include "includes.h"
__global__ void calculate_A_ch_1_2(float* rho, float* dz, float* s_a, float* xx_or_yy, float* s_b, float K, int npix, int nchannels, int nimages, float* A_ch) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
int c = blockIdx.z*blockDim.z + threadIdx.z;
if (i < npix && j < nimages) {
A_ch[c*npix*nimages + j*npix + i] = (rho[c*npix + i] / dz[i])*(K*s_a[c * nimages * 3 + j] - xx_or_yy[i] * s_b[c * nimages * 3 + j]);
}
}