#include "includes.h"
__device__ int sign(DECNUM x)
{
return((x > 0.0f) - (x < 0.0f));
}
__device__ int mminus2(int ix, int nx)
{
int xminus;
if (ix <= 1)
{
xminus = 0;
}
else
{
xminus = ix - 2;
}
return(xminus);
}
__device__ int pplus(int ix, int nx)
{
int xplus;
if (ix == nx - 1)
{
xplus = nx - 1;
}
else
{
xplus = ix + 1;
}
return(xplus);

}
__device__ int mminus(int ix, int nx)
{
int xminus;
if (ix == 0)
{
xminus = 0;
}
else
{
xminus = ix - 1;
}
return(xminus);
}
__global__ void latbnd(int nx, int ny, DECNUM * uu)
{
unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
unsigned int i = ix + iy*nx;
int tx = threadIdx.x;
int ty = threadIdx.y;

__shared__ DECNUM uut[16][16];
__shared__ DECNUM uub[16][16];

if (ix < nx && iy < ny)
{
unsigned int yminus = mminus(iy, ny);
unsigned int yminus2 = mminus2(iy, ny);
unsigned int yplus = pplus(iy, ny);


uut[tx][ty] = uu[ix + yplus*nx];
uub[tx][ty] = uu[ix + yminus*nx];

if (iy == 0)
{
uu[i] = uut[tx][ty];
}
if (iy == ny - 1)
{
uu[i] = uub[tx][ty];
}
}
//
}