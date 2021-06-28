#include "includes.h"
__global__ void returnStatistic ( const int dim, const int nwl, const float *xx, float *s ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * dim;
if ( i < dim && j < nwl ) {
s[t] = powf ( xx[t], 2. );
}
}