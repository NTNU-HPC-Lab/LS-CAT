#include "includes.h"
__global__ void SortDistances(float* dist, int* idMat, int n, int k)
{
// Get the index of the column that the current thread is responsible for
auto col = blockIdx.x * blockDim.x + threadIdx.x;

// IF col is out of bounds, then do nothing
if(col < n)
{
auto id = &idMat[col * n];
for(auto i = 0; i < n; ++i)
id[i] = i;

auto distCol = &dist[col * n];
// Only care about the first k elements being sorted
for (auto i = 0; i < k; ++i)
{
auto minIndex = i;
for (auto j = i + 1; j < n; ++j)
{
if(distCol[j] < distCol[minIndex])
minIndex = j;
}
auto tmp = distCol[minIndex];
distCol[minIndex] = distCol[i];
distCol[i] = tmp;

auto tmpId = id[minIndex];
id[minIndex] = id[i];
id[i] = tmpId;
}
}
}