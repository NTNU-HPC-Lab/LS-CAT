#include "includes.h"
__global__ void kernel4(int k, int n, int bias, float* searchPoints, float* referencePoints, float* dist)
{
float diff, squareSum;
int tid = blockDim.x * blockIdx.x + threadIdx.x;
if (tid < n) {
squareSum = 0;
for (int i = 0; i < k; i++) {
diff = searchPoints[k * bias + i] - referencePoints[k * tid + i];
squareSum += (diff * diff);
}
dist[bias * n + tid] = squareSum;
}
}