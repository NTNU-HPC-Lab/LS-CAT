#include "includes.h"
__global__ void arr_times_const_checkerboard(float*a,float b, float * c, int N, int sx,int sy,int sz)
{
int ids=blockIdx.x*blockDim.x+threadIdx.x;   // which source array element do I have to deal with?
if(ids>=N) return;  // not in range ... quit

int px=(ids/2)%sx;   // my x pos
int py=(ids/2)/sx;   // my y pos
float minus1=(1-2*((px+py)%2));
c[ids]=a[ids]*b*minus1;
}