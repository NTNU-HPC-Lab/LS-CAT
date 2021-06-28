#include "includes.h"
__global__ void PositiveDefiniteKernel( char *hessian_pd, float *hessian, int imageW, int imageH, int imageD )
{
const int baseX = blockIdx.x * PD_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * PD_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z * PD_BLOCKDIM_Z + threadIdx.z;
const int size = imageW * imageH * imageD;
const int idx = (baseZ * imageH + baseY) * imageW + baseX;

float xx = hessian[idx];
float xy = hessian[idx + size];
float xz = hessian[idx + size*2];
float yy = hessian[idx + size*3];
float yz = hessian[idx + size*4];
float zz = hessian[idx + size*5];

// Sylvester's criterion
hessian_pd[idx] = (
xx < 0 &&
xx*yy-xy*xy > 0 &&
xx*yy*zz + 2*xy*yz*xz - xx*yz*yz - yy*xz*xz - zz*xy*xy < 0
) ? 1 : 0;

}