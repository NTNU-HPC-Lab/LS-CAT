#include "includes.h"
__global__ void zupdate_stencil(float *zx, float *zy, float *zoutx, float *zouty, float *g, float tau, float invlambda, int  nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;
int tidx, tpx, tpy;
float a, b, t;
float DIVZ;

/* compute simultaneously
f= div z -g /lambda at the positions
right, center north*/
float fr = 0, fc = 0, fu = 0;

////////////////////////////////////////////////////////
//
//		(zul)		(zu)
//
//					____
//		(zl)		|zc|		(zr)
//					----
//
//					(zd)		(zdr)
//
//		if the pixel is not inside the region then put 0
//
//		fc = z1c - z1l + z2c - z2d
//		fr = z1r - z1c + z2r - z2dr
//		fu = z1u - z1ul + z2u - z2c
//
////////////////////////////////////////////////////////

tidx = idx;
tpx = px;
tpy = py;
if (tpx<nx && tpy<ny)
{
// compute the divergence
DIVZ = 0;
if ((tpx<(nx - 1))) DIVZ += zx[tidx];
if ((tpx>0))      DIVZ -= zx[tidx - 1];

if ((tpy<(ny - 1))) DIVZ += zy[tidx];
if ((tpy>0))      DIVZ -= zy[tidx - nx];

fc = DIVZ;
}
////////////////////////////////////////////////////////

tidx = idx + 1;
tpx = px + 1;
tpy = py;
if (tpx<nx && tpy<ny)
{
// compute the divergence
DIVZ = 0;
if ((tpx<(nx - 1))) DIVZ += zx[tidx];
if ((tpx>0))      DIVZ -= zx[tidx - 1];

if ((tpy<(ny - 1))) DIVZ += zy[tidx];
if ((tpy>0))      DIVZ -= zy[tidx - nx];

fr = DIVZ;
}
////////////////////////////////////////////////////////

tidx = idx + nx;
tpx = px;
tpy = py + 1;
if (tpx<nx && tpy<ny)
{
// compute the divergence
DIVZ = 0;
if ((tpx<(nx - 1))) DIVZ += zx[tidx];
if ((tpx>0))      DIVZ -= zx[tidx - 1];

if ((tpy<(ny - 1))) DIVZ += zy[tidx];
if ((tpy>0))      DIVZ -= zy[tidx - nx];

fu = DIVZ;
}

fr = fr - g[idx + 1] * invlambda;
fc = fc - g[idx] * invlambda;
fu = fu - g[idx + nx] * invlambda;

////////////////////////////////////////////////////////

if (px<nx && py<ny)
{
// compute the gradient
a = 0;
b = 0;
if (px<(nx - 1)) a = fr - fc;
if (py<(ny - 1)) b = fu - fc;

// update z
t = 1 / (1 + tau*sqrtf(a*a + b*b));
zoutx[idx] = (zx[idx] + tau*a)*t;
zouty[idx] = (zy[idx] + tau*b)*t;
}
}