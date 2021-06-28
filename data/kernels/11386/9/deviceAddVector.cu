#include "includes.h"
__global__ void deviceAddVector(int *d_a, int *d_b, int *d_c, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size) {
d_c[i] = d_a[i] + d_b[i];
//  printf("Tread %d make sum %d + %d = %d", i, d_a[i], d_b[i], d_c[i]);
}
}