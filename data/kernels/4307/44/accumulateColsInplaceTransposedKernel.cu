#include "includes.h"
__global__ void accumulateColsInplaceTransposedKernel(float *input, int channels, int h, int w) {
// in-place.
// input is a `(w+1) x channels * (h+1)` array

// global column index (of all `channels * w` columns in this image)
int colIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

if (colIdx < channels * h) {
// need to zero the (0,0) corner of the output separately >:(
input[(colIdx / h) * (h+1)] = 0;

colIdx += colIdx / h + 1; // make `colIdx` the (h+1)-array indexer

input[colIdx] = 0; // first element of every column is always zero

double sum = 0;

for (int i = 1; i <= w; ++i) {
float *currentElement = &input[i * channels * (h+1) + colIdx];
sum += static_cast<double>(*currentElement);
*currentElement = static_cast<float>(sum);
}
}
}