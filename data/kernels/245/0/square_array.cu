#include "includes.h"
__global__ void square_array(double *a, int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx<N) a[idx] = a[idx] * a[idx];
printf("idx = %d, a = %f\n", idx, a[idx]);
}