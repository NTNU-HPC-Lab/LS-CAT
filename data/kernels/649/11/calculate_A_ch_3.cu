#include "includes.h"
__global__ void calculate_A_ch_3(float* rho, float* dz, float* s_a, int npix, int nchannels, int nimages, float* A_ch) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
int c = blockIdx.z*blockDim.z + threadIdx.z;
if (i < npix && j < nimages) {
A_ch[c*npix*nimages + j*npix + i] = (rho[c*npix + i] / dz[i])*(s_a[c * nimages * 3 + j]);
}
}