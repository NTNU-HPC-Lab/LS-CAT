#include "includes.h"
__global__ void returnQM ( const int dim, const int n, const float *s1, const float *s0, float *q ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
q[i] = expf ( - 0.5 * ( s1[i] - s0[i] ) );
}
}