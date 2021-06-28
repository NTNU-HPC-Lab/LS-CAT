#include "includes.h"
__global__ void accumulateRowsKernel( float *input, float *output, int channels, int h, int w) {
// view multichannel image as a multiline single-channel image
int globalRowIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

if (globalRowIdx < channels * h) {
float *outputRow = output + (globalRowIdx + globalRowIdx / h + 1) * (w+1) + 1;
outputRow[-1] = 0;

double sum = 0;
for (int i = 0; i < w; ++i) {
sum += input[globalRowIdx * w + i];
outputRow[i] = static_cast<float>(sum);
}

// need to zero the (0,0) corner of the output separately >:(
output[(globalRowIdx / h) * (w+1) * (h+1)] = 0;
}
}