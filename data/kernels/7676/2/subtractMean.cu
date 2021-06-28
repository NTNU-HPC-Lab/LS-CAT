#include "includes.h"



// helper for CUDA error handling
__global__ void subtractMean( double* images, const double* meanImage, std::size_t imageNum, std::size_t pixelNum )
{
std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

if(col >= pixelNum)
{
return;
}

for(std::size_t row = 0; row < imageNum; ++row)
{
images[row*pixelNum + col] -= meanImage[col];

if(images[row*pixelNum + col] < 0.0)
{
images[row*pixelNum + col] = 0.0;
}
}
}