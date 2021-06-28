#include "includes.h"
__global__ void accumulateColsKernel(float *input, float *output, int channels, int h, int w) {
// global column index (of all `channels * w` columns in this image)
int colIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

if (colIdx < channels * w) {
// jump to current channel
input  += (colIdx / w) * h * w;
output += (colIdx / w) * (h+1) * (w+1);
colIdx %= w; // switch to local column index,
++colIdx;    // it's 1-indexed because first output column is always zero

output[colIdx] = 0; // first element of every column is always zero
double sum = 0;

for (int i = 1; i <= h; ++i) {
sum += static_cast<double>(input[(i-1) * w + colIdx - 1]);
output[i * (w+1) + colIdx] = static_cast<float>(sum);
}
}
}