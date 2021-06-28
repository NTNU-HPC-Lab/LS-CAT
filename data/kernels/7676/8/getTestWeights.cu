#include "includes.h"



// helper for CUDA error handling
__global__ void getTestWeights( const double* restoredEigenvectors , const double* meanImage , const double* testImages , double* testWeights , std::size_t testImageNum , std::size_t pixelNum , std::size_t componentNum )
{
std::size_t row = blockIdx.x;
std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;

if(col >= testImageNum || row >= componentNum)
{
return;
}

testWeights[row * testImageNum + col] = 0.0;
for(std::size_t i = 0; i < pixelNum; ++i)
{
double testImagePixelValue = testImages[col * pixelNum + i] - meanImage[i];
if(testImagePixelValue < 0.0)
{
testImagePixelValue = 0.0;
}
testWeights[row * testImageNum + col] += restoredEigenvectors[row * pixelNum + i] * (testImagePixelValue);
}
}