#include "includes.h"
__global__ void InsertionSortDistances(float* dist, int* idMat, int n, int k)
{
// Get the index of the column that the current thread is responsible for
auto col = blockIdx.x * blockDim.x + threadIdx.x;

// IF col is out of bounds, then do nothing
if (col < n)
{
auto id = &idMat[col * n];

id[0] = 0;

auto distCol = &dist[col * n];

// Otherwise, sort column 'col'
auto i = 1;
while(i < n)
{
auto x = distCol[i];
auto currIndex = i;
auto j = i - 1;
while(j >= 0 && distCol[j] > x)
{
distCol[j + 1] = distCol[j];
id[j + 1] = id[j];
--j;
}
distCol[j + 1] = x;
id[j + 1] = currIndex;
++i;
}
}
}