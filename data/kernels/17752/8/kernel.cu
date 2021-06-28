#include "includes.h"
__global__ void kernel(int k, int n, float* sub_searchPoints, float* referencePoints, float* dist)
{
float diff, squareSum;
int tid = blockDim.x * blockIdx.x + threadIdx.x;
if (tid < n) {
squareSum = 0;
for (int i = 0; i < k; i++) {
diff = sub_searchPoints[i] - referencePoints[k * tid + i];
squareSum += (diff * diff);
}
dist[tid] = squareSum;
}
}