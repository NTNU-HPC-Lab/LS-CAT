#include "includes.h"
__global__ void mapRandomNumbers ( const int nwl, const int ist, const int isb, const float *r, float *zr, int *kr, float *ru, int *kex ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int rr;
if ( i < nwl ) {
rr = i + 0 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
zr[i] = 1. / ACONST * powf ( r[rr] * ( ACONST - 1 ) + 1, 2. );
rr = i + 1 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
kr[i] = ( int ) truncf ( r[rr] * ( nwl - 1 + 0.999999 ) );
rr = i + 2 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
ru[i] = r[rr];
rr = i + 3 * nwl + isb * 4 * nwl + ist * 4 * 2 * nwl;
kex[i] = ( int ) truncf ( r[rr] * ( 5 - 1 + 0.999999 ) );
}
}