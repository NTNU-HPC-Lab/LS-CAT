#include "includes.h"



// helper for CUDA error handling
__global__ void restoreEigenvectors( const double* meanSubtractedImages , const double* reducedEigenvectors , double* restoredEigenvectors , std::size_t imageNum , std::size_t pixelNum , std::size_t componentNum )
{
std::size_t row = blockIdx.x;
std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;

if(col >= pixelNum || row >= componentNum)
{
return;
}

restoredEigenvectors[row * pixelNum + col] = 0.0;
for(std::size_t i = 0; i < imageNum; ++i)
{
restoredEigenvectors[row * pixelNum + col] += reducedEigenvectors[(imageNum - row - 1) * imageNum + i] * meanSubtractedImages[i * pixelNum + col];
}
}