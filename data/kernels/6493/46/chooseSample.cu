#include "includes.h"
__global__ void chooseSample ( const int nDB, const int si, const float *EE, float *EBV ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nDB ) {
EBV[i] = EE[i+si*nDB];
}
}