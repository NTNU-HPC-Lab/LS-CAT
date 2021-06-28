#include "includes.h"
__global__ void inc (int n, float* a) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
a[i] += 1;
}
};