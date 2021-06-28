#include "includes.h"
__global__ void worker(double * a, long n) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < n) {
a[i] += i;
}
}