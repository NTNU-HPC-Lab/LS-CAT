#include "includes.h"



__global__ void BlockPrefix(int *a, int k, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
for(int j = i * k + 1; j < i * k + k && j < n; ++j) {
a[j] += a[j - 1];
}
}