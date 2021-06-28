#include "includes.h"
__global__ void returnQ ( const int dim, const int n, const float *s1, const float *s0, const float *zr, float *q ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
q[i] = expf ( - 0.5 * ( s1[i] - s0[i] ) ) * powf ( zr[i], dim - 1 );
}
}