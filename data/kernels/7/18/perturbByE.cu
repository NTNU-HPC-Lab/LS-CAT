#include "includes.h"
__global__ void perturbByE( float *tmppos, float4 *mypos, float eps, float *E, float *masses, int k, int m, int N ) {
int dof = blockIdx.x * blockDim.x + threadIdx.x;
if( dof >= N ) {
return;
}
int atom = dof / 3;

int axis = dof % 3;
if( axis == 0 ) {
tmppos[dof] = mypos[atom].x;
mypos[atom].x += eps * E[dof * m + k] / sqrt( masses[atom] );
} else if( axis == 1 ) {
tmppos[dof] = mypos[atom].y;
mypos[atom].y += eps * E[dof * m + k] / sqrt( masses[atom] );
} else {
tmppos[dof] = mypos[atom].z;
mypos[atom].z += eps * E[dof * m + k] / sqrt( masses[atom] );
}
}