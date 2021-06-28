#include "includes.h"
__global__ void returnQ1 ( const int dim, const int n, const float *p1, const float *p0, const float *s1, const float *s0, const float *zr, float *q ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
if ( p1[i] == INF || - 0.5 * ( s1[i] + p1[i] - s0[i] - p0[i] ) < -10. ) {
q[i] = 0.0;
} else if ( - 0.5 * ( s1[i] + p1[i] - s0[i] - p0[i] ) > 10. ) {
q[i] = 1.E10;
} else {
q[i] = expf ( - 0.5 * ( s1[i] + p1[i] - s0[i] - p0[i] ) ) * powf ( zr[i], dim - 1 );
}
}
}