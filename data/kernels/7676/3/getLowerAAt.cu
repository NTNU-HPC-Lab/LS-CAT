#include "includes.h"



// helper for CUDA error handling
__global__ void getLowerAAt( const double* A, double* S, std::size_t imageNum, std::size_t pixelNum )
{
std::size_t row = blockIdx.x;
std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;

if(row >= imageNum || col >= imageNum)
{
return;
}

S[row * imageNum + col] = 0.0;
for(std::size_t i = 0; i < pixelNum; ++i)
{
S[row * imageNum + col] += A[row * pixelNum + i] * A[col * pixelNum + i];
}
}