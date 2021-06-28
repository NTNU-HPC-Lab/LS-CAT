#include "includes.h"

#define N 18



__global__ void sum(double *a, double *b, double *c) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
c[index] = a[index] + b[index];
}