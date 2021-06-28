#include "includes.h"
__global__ void device_add(int *a, int *b, int *c) {

c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}