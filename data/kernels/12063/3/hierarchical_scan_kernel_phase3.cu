#include "includes.h"
__global__ void hierarchical_scan_kernel_phase3(int *S, int *Y) {

int tx = threadIdx.x, bx = blockIdx.x;
int i = bx * SECTION_SIZE + tx;
//printf("Y[%d] = %.2f\n", i, Y[i]);

if (bx > 0)
{
for (int j = 0; j < SECTION_SIZE ; j += BLOCK_DIM ) {
Y[i + j] += S[bx - 1];
}
}
}