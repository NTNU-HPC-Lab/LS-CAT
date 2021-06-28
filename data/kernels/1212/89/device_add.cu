#include "includes.h"
__global__ void device_add(int *a, int *b, int *c) {

c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}