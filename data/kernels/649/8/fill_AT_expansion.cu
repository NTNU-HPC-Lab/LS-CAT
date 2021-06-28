#include "includes.h"
__global__ void fill_AT_expansion(float* A, int* rowind, int* colind, float* val, int npix, int nimages) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < npix*nimages) {
colind[i] = i / nimages + (i % nimages)*npix;
rowind[i] = i / nimages;
val[i] = A[colind[i]];
}
}