#include "includes.h"
__global__ void lhkde ( const int n, const float *a, const float *b, float *l, float *h  ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
l[i] = a[i] - 3 * b[i];
h[i] = a[i] + 3 * b[i];
}
}