#include "includes.h"
__global__ void fill_A_expansion(float* A, int* rowind, int* colind, float* val, int npix, int nimages) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < npix*nimages) {
rowind[i] = i;
colind[i] = i % npix;
val[i] = A[i];
}
}