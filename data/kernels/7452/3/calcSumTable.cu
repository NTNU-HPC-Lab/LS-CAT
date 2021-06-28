#include "includes.h"
__global__ void calcSumTable(const float *rowCumSum, float *SumTable, int rowNumberN, int colNumberM) {
int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
if (xIndex >= colNumberM) return;
for (int i = 1; i < rowNumberN; i++) {
SumTable[i * colNumberM + xIndex] +=
rowCumSum[(i - 1) * colNumberM + xIndex];
}
}