#include "includes.h"
__global__ void compute_B_for_depth(float* B, float* rho, float* Ns, int npix, int nchannels, int nimages) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int c = blockIdx.y*blockDim.y + threadIdx.y;
if (i < npix*nimages) {
B[c*npix*nimages + i] -= rho[c*npix + i%npix] * Ns[c*npix*nimages + i];
}
}