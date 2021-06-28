#include "includes.h"
__global__ void init(int order, const int matrices, double * C)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

for (int b=0; b<matrices; ++b) {
if ((i<order) && (j<order)) {
C[b*order*order+i*order+j] = 0;
}
}
}