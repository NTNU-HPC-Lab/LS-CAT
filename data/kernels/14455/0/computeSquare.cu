#include "includes.h"



__global__ void computeSquare(int *d_in, int *d_out) {
int index = threadIdx.x;
d_out[index] = d_in[index] * d_in[index];
}