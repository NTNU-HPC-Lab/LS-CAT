#include "includes.h"
__global__ void ComputeAdjacencyMatrix(float* dOut, int* nn, int n, int k)
{
// Get the column that the current thread is responsible for
auto col = blockIdx.x * blockDim.x + threadIdx.x;

// If id is within bounds
if(col < n)
{
auto nnCol = &nn[col * n];
for(auto i = 0; i < k; ++i)
{
dOut[col * n + nnCol[i]] = dOut[col + n * nnCol[i]] = 1.0f;
}
}
}