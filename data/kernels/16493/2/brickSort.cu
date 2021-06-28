#include "includes.h"
__global__ void brickSort(int* array, int arrayLen, int p) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= arrayLen - 1)
return;
if ((p % 2 == 0) && (idx % 2 == 1))
return;
if ((p % 2 == 1) && (idx % 2 == 0))
return;
if (array[idx] > array[idx + 1]) {
int tmp = array[idx + 1];
array[idx + 1] = array[idx];
array[idx] = tmp;
}
}