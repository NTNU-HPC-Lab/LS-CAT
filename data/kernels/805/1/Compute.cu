#include "includes.h"



__global__ void Compute(int *a, int k, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int id = i / k * 2 * k + k + i % k;
if(id < n) {
a[id] += a[id - id % k - 1];
}
}