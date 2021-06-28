#include "includes.h"
__global__ void SPkernel(int k, int m, int n, float* searchPoints, float* referencePoints, int* indices)
{
int minIndex;
float minSquareSum, diff, squareSum;
int tid = blockDim.x * blockIdx.x + threadIdx.x;
if (tid < m) {
minSquareSum = -1;
// Iterate over all reference points
for (int nInd = 0; nInd < n; nInd++) {
squareSum = 0;
for (int kInd = 0; kInd < k; kInd++) {
diff = searchPoints[k * tid + kInd] - referencePoints[k * nInd + kInd];
squareSum += (diff * diff);
}
if (minSquareSum < 0 || squareSum < minSquareSum) {
minSquareSum = squareSum;
minIndex = nInd;
}
}
indices[tid] = minIndex;
}
}