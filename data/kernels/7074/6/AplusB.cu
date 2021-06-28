#include "includes.h"
__global__ void AplusB(int *ret, int a, int b) {
ret[threadIdx.x] = a + b + threadIdx.x;
}