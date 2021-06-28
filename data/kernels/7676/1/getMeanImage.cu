#include "includes.h"



// helper for CUDA error handling
__global__ void getMeanImage( const double* images, double* meanImage, std::size_t imageNum, std::size_t pixelNum )
{
std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

if(col >= pixelNum)
{
return;
}

meanImage[col] = 0.0;
for(std::size_t row = 0; row < imageNum; ++row)
{
meanImage[col] += images[row*pixelNum + col];
}

meanImage[col] /= imageNum;
}