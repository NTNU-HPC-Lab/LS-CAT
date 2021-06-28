#include "includes.h"



// helper for CUDA error handling
__global__ void getWeights( const double* restoredEigenvectors , const double* meanSubtractedImages , double* weights , std::size_t imageNum , std::size_t pixelNum , std::size_t componentNum )
{
std::size_t row = blockIdx.x;
std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;

if(col >= imageNum || row >= componentNum)
{
return;
}

weights[row * imageNum + col] = 0.0;
for(std::size_t i = 0; i < pixelNum; ++i)
{
weights[row * imageNum + col] += restoredEigenvectors[row * pixelNum + i] * meanSubtractedImages[col * pixelNum + i];
}
}