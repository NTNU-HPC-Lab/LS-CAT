#include "includes.h"
__global__ void ComputeLaplacianInPlace(float* d, int n)
{
// Column to sum
auto x = blockIdx.x * blockDim.x + threadIdx.x;

if(x < n)
{
auto dCol = &d[x * n];

for(auto i = 0; i < n; ++i)
{
if(i != x)
{
dCol[x] += dCol[i];
dCol[i] = -dCol[i];
}
}
}
}