#include "includes.h"
__global__ void sumaVectores (float * d_a, float *d_b, float * d_c) {

int index = blockIdx.x*blockDim.x+threadIdx.x;
if (index < N )
d_c[index] = d_a[index] +d_b[index];
}