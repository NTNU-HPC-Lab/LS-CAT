#include "includes.h"
__global__ void ScaleDownDense(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
#define BW (SCALEDOWN_W+4)
#define BH (SCALEDOWN_H+4)
#define W2 (SCALEDOWN_W/2)
#define H2 (SCALEDOWN_H/2)
__shared__ float irows[BH*BW];
__shared__ float brows[BH*W2];
const int tx = threadIdx.x;
const int ty = threadIdx.y;
const int xp = blockIdx.x*SCALEDOWN_W + tx;
const int yp = blockIdx.y*SCALEDOWN_H + ty;
const int xl = min(width-1,  max(0, xp-2));
const int yl = min(height-1, max(0, yp-2));
const float k0 = d_ScaleDownKernel[0];
const float k1 = d_ScaleDownKernel[1];
const float k2 = d_ScaleDownKernel[2];
if (xp<(width+4) && yp<(height+4))
irows[BW*ty + tx] = d_Data[yl*pitch + xl];
__syncthreads();
if (yp<(height+4) && tx<W2) {
float *ptr = &irows[BW*ty + 2*tx];
brows[W2*ty + tx] = k0*(ptr[0] + ptr[4]) + k1*(ptr[1] + ptr[3]) + k2*ptr[2];
}
__syncthreads();
const int xs = blockIdx.x*W2 + tx;
const int ys = blockIdx.y*H2 + ty;
if (tx<W2 && ty<H2 && xs<(width/2) && ys<(height/2)) {
float *ptr = &brows[W2*(ty*2) + tx];
d_Result[ys*newpitch + xs] = k0*(ptr[0] + ptr[4*W2]) + k1*(ptr[1*W2] + ptr[3*W2]) + k2*ptr[2*W2];
}
}