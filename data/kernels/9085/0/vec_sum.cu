#include "includes.h"

#define N 100000000


__global__ void vec_sum(float* a, float* b, float* c) {
int bid = blockIdx.x;
if (bid < N) {
c[bid] = a[bid] + b[bid];
}
}