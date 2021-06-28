#include "includes.h"
__global__ void rotate2(float*a,float b, float * c, int sx,int sy,int sz, int dx, int dy, int dz)
{
int ids=(blockIdx.x*blockDim.x+threadIdx.x); // id of this processor
int x=(ids + dx)%sx;  // advance by the offset steps along the chain
int y=(ids/sx + dy)%sy;
int z=(ids/(sx*sy) + dz)%sz;
int idd=x+sx*y+sx*sy*z;
if(ids>=sx*sy*sz) return;
// float tmp = a[ids];
// __syncthreads();             // nice try but does not work !
c[idd] = b*a[ids];
}