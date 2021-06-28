#include "includes.h"
__global__ void scatter(unsigned int *d_inVals, unsigned int *d_outVals, unsigned int *d_inPos, unsigned int *d_outPos, unsigned int *d_zerosScan, unsigned int *d_onesScan, unsigned int *d_zerosPredicate, unsigned int *d_onesPredicate, size_t n)
{
int tx = threadIdx.x;
int bx = blockIdx.x;
int index = BLOCK_WIDTH * bx + tx;
int offset = d_zerosScan[n - 1] + d_zerosPredicate[n - 1];

if(index < n) {
int scatterIdx;
if(d_zerosPredicate[index]) {
scatterIdx = d_zerosScan[index];
} else {
scatterIdx = d_onesScan[index] + offset;
}
if(scatterIdx < n) { //sanity check
d_outVals[scatterIdx] = d_inVals[index];
d_outPos[scatterIdx] = d_inPos[index];
}
}
}