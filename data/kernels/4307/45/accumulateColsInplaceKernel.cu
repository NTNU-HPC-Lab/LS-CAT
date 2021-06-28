#include "includes.h"
__global__ void accumulateColsInplaceKernel(float *input, int channels, int h, int w) {
// in-place.
// input is already a `channels * (h+1) x (w+1)` array

// global column index (of all `channels * w` columns in this image)
int colIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

if (colIdx < channels * w) {
input += (colIdx / w) * (h+1) * (w+1); // jump to current channel
colIdx %= w; // switch to local column index,
++colIdx;    // it's 1-indexed because first output column is always zero

input[colIdx] = 0; // first element of every column is always zero
double sum = 0;

for (int i = 1; i <= h; ++i) {
float *currentElement = &input[i * (w+1) + colIdx];
sum += static_cast<double>(*currentElement);
*currentElement = static_cast<float>(sum);
}
}
}