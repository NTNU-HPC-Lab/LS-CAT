#include "includes.h"
__global__ void updateStatistic ( const int nwl, const float *stt1, const float *q, const float *r, float *stt0 ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nwl ) {
stt0[i] = ( q[i] > r[i] ) * stt1[i] + ( q[i] < r[i] ) * stt0[i];
}
}