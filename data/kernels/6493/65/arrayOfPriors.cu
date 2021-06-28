#include "includes.h"
__global__ void arrayOfPriors ( const int dim, const int nwl, const float *cn, const float *xx, float *pr ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
float sum = 0.;
if ( i < nwl ) {
pr[i] = ( cn[i] == dim ) * sum + ( cn[i] < dim ) * INF;
}
}