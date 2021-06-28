#include "includes.h"
__global__ void addScannedBlockSums(float *input, float *aux, int len) {

int tx = threadIdx.x;

int bx = blockIdx.x;

int dx = blockDim.x;

int i = 2 * bx * dx + tx;

if (bx > 0) {

if (i < len)
aux[i] += input[bx-1];

if (i + dx < len)
aux[i + dx] += input[blockIdx.x - 1];
}
}