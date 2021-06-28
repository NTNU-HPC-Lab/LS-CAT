#include "includes.h"
__global__ void cuda_conv2D_backPropError(double* upStreamError, const double* err, const double* pKernels, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputChannels, size_t inputRows, size_t inputCols, size_t padding, size_t stride)
{
// Do all values for i, j, and k in parallel
int id = blockIdx.x * blockDim.x + threadIdx.x;
size_t i = id % inputChannels;
id /= inputChannels;
size_t j = id % outputRows;
id /= outputRows;
if(id >= kernelCount)
return;
size_t k = id;

// Compute some intermediate values
size_t outChannelOffset = k * outputRows * outputCols;
size_t outRowOffset = j * outputCols;
int inRowOffset = j * stride - padding;

// This block of code is derived from the serial implementation
size_t kk = k * inputChannels * kernelRows * kernelCols;
size_t index = outChannelOffset + outRowOffset + i;
int inColOffset = i * stride - padding;
for(size_t z = 0; z < inputChannels; z++)
{
size_t kernelChannelOffset = z * kernelRows * kernelCols;
size_t inChannelOffset = z * inputRows * inputCols;
for(size_t y = 0; y < kernelRows; y++)
{
size_t kernelRowOffset = y * kernelCols;
int inRow = inRowOffset + y;
for(size_t x = 0; x < kernelCols; x++)
{
int inCol = inColOffset + x;
if(inRow >= 0 && inRow < (int)inputRows && inCol >= 0 && inCol < (int)inputRows)
{
size_t idx = inChannelOffset + inputCols * inRow + inCol;
upStreamError[idx] += pKernels[kk + kernelChannelOffset + kernelRowOffset + x] * err[index];
}
}
}
}

}