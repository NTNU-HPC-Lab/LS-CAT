#include "includes.h"
__global__ void symmetrize2D( float *h, int natoms ) {
const int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
const int dof = 3 * natoms;
if( elementNum >= dof * dof ) {
return;
}
int r = elementNum / dof;
int c = elementNum % dof;

if( r > c ) {
return;
} else {
const float avg = 0.5 * ( h[r * dof + c] + h[c * dof + r] );
h[r * dof + c] = avg;
h[c * dof + r] = avg;
}
}