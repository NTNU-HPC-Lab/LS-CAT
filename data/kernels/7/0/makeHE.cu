#include "includes.h"
__global__ void makeHE( float *HE, float *force1, float4 *force2, float *masses, float eps, int k, int m, int N ) {
int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
int atom = elementNum / 3;
if( elementNum >= N ) {
return;
}

int axis = elementNum % 3;
if( axis == 0 ) {
HE[elementNum * m + k] = ( force1[elementNum] - force2[atom].x ) / ( sqrt( masses[atom] ) * 1.0 * eps );
} else if( axis == 1 ) {
HE[elementNum * m + k] = ( force1[elementNum] - force2[atom].y ) / ( sqrt( masses[atom] ) * 1.0 * eps );
} else {
HE[elementNum * m + k] = ( force1[elementNum] - force2[atom].z ) / ( sqrt( masses[atom] ) * 1.0 * eps );
}
}