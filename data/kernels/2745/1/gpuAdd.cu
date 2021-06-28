#include "includes.h"
__global__ void gpuAdd(int d_a, int d_b, int *d_c) {
*d_c = d_a + d_b;
}