#include "includes.h"
__global__ void powWalkers ( const int n, const float c, const float *a, float *d ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
d[i] = powf ( a[i], c );
}
}