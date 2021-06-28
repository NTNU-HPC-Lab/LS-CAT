#include "includes.h"
__global__ void addArraysGPU(int* a, int* b, int* c) {
int i = threadIdx.x;
c[i] = a[i] + b[i];
}