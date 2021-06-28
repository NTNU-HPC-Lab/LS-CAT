#include "includes.h"
__global__ void AssembleArrayOfAbsorptionFactors ( const int nmbrOfWlkrs, const int nmbrOfEnrgChnnls, const int nmbrOfElmnts, const float *crssctns, const float *abndncs, const int *atmcNmbrs, const float *wlkrs, float *absrptnFctrs ) {
int enIndx = threadIdx.x + blockDim.x * blockIdx.x;
int wlIndx = threadIdx.y + blockDim.y * blockIdx.y;
int ttIndx = enIndx + wlIndx * nmbrOfEnrgChnnls;
int elIndx, effElIndx, crIndx, prIndx;
float xsctn, clmn, nh;
if ( enIndx < nmbrOfEnrgChnnls && wlIndx < nmbrOfWlkrs ) {
if ( NHINDX == NPRS-1 ) {
elIndx = 0;
prIndx = elIndx + NHINDX;
crIndx = elIndx + enIndx * nmbrOfElmnts;
effElIndx = atmcNmbrs[elIndx] - 1;
nh = wlkrs[prIndx+wlIndx*NPRS] * 1.E22;
clmn = abndncs[effElIndx];
xsctn = clmn * crssctns[crIndx];
elIndx = 1;
while ( elIndx < nmbrOfElmnts ) {
prIndx = elIndx + NHINDX;
crIndx = elIndx + enIndx * nmbrOfElmnts;
effElIndx = atmcNmbrs[elIndx] - 1;
clmn = abndncs[effElIndx]; // * powf ( 10, wlkrs[wlIndx].par[prIndx] );
xsctn = xsctn + clmn * crssctns[crIndx];
elIndx += 1;
}
absrptnFctrs[ttIndx] = expf ( - nh * xsctn );
} else if ( NHINDX == NPRS ) {
absrptnFctrs[ttIndx] = 1;
}
}
}