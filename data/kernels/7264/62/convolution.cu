#include "includes.h"
__global__ void convolution(float* input, int inputRows, int inputCols, int inputLd, float* kernel, int kernelRows, int kernelCols, int kernelLd, int rowStep, int colStep, float* output, int outputLd) {

int row = (blockIdx.y * blockDim.y + threadIdx.y) * rowStep;
int col = (blockIdx.x * blockDim.x + threadIdx.x) * colStep;

if (row <= inputRows - kernelRows && col <= inputCols - kernelCols) {
int i, j;
output[row+col*outputLd] = 0;
for (i=0; i<kernelRows; i++) {
for (j=0; j<kernelCols; j++) {
output[row+col*outputLd] += kernel[i+j*kernelLd] * input[(row+i)+(col+j)*inputLd];
}
}
}

}