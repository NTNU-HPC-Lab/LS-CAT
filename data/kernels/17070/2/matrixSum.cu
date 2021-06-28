#include "includes.h"
__global__ void matrixSum(int* a, int* b, int* c, int size)
{
// printf("ERROR en global\n");
int pos = threadIdx.x;
if (pos < size * size) {
c[pos] = a[pos] + b[pos];
}
}