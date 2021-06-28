#include "includes.h"
__global__ void arrayOfPriors1 ( const int dim, const int nwl, const float *cn, const float *nhMd, const float *nhSg, const float *xx, float *pr ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
float sum; //, theta, kk;
if ( i < nwl ) {
//theta = powf ( nhSg[i], 2 ) / nhMd[i];
//kk = nhMd[i] / theta;
//sum = ( kk - 1 ) * logf ( xx[NHINDX+i*nwl] ) - xx[NHINDX+i*nwl] / theta;
sum = 0; //powf ( ( xx[NHINDX+i*nwl] - nhMd[i] ) / nhSg[i], 2 );
pr[i] = ( cn[i] == dim ) * sum + ( cn[i] < dim ) * INF;
}
}