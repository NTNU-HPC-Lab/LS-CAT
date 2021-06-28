#include "includes.h"



// helper for CUDA error handling
__global__ void getDistances( const double* trainingWeights , const double* testWeights , double* distances , std::size_t trainImageNum , std::size_t testImageNum , std::size_t componentNum )
{
std::size_t row = blockIdx.x;
std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;

if(col >= testImageNum || row >= trainImageNum)
{
return;
}

distances[row * testImageNum + col] = 0.0;
for(std::size_t i = 0; i < componentNum; ++i)
{
distances[row * testImageNum + col] += fabs(trainingWeights[i * trainImageNum + row] - testWeights[i * testImageNum + col]);
}
}