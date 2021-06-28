#include "includes.h"
# pragma warning (disable:4819)



#define ARRAYSIZE 5

__global__ void addKernel(int *c, const int *a, const int *b) {
int i = threadIdx.x;
c[i] = a[i] + b[i];
}