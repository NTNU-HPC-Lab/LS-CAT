#include "includes.h"
__global__ void addKernel(int* c, const int* a, const int* b, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
// since we're asking for one more thread than elements in the arrays
// we need to handle size to make sure we don't access beyond the end of the array
if (i < size) {
c[i] = a[i] + b[i];
}
}