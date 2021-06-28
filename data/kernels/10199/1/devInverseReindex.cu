#include "includes.h"
__global__ void devInverseReindex(int N, int *destArray, int *srcArray, unsigned int *reindex, int realSize, int nDims, int maxValue, bool ignoreValue)
{
for (unsigned int n = 0; n < nDims; n++) {
int i = blockIdx.x*blockDim.x + threadIdx.x;

while (i < N) {
int ret = -1;
int tmp = srcArray[i + n*realSize];
if (tmp != -1 || ignoreValue == false) {
int addValue = 0;
while (tmp >= maxValue) {
tmp -= maxValue;
addValue += maxValue;
}
while (tmp < 0) {
tmp += maxValue;
addValue -= maxValue;
}
ret = (int) reindex[tmp] + addValue;
}
destArray[i + n*realSize] = ret;

i += gridDim.x*blockDim.x;
}
}
}