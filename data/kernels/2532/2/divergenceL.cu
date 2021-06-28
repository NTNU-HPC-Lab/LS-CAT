#include "includes.h"
__global__ void divergenceL(float *v, float *d, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;

/*
float AX = 0;
if ((idx < N) && (px<(nx-1))) AX += v[2*(idx   )+0];
if ((idx < N) && (px>0))      AX -= v[2*(idx-1 )+0];

if ((idx < N) && (py<(ny-1))) AX += v[2*(idx   )+1];
if ((idx < N) && (py>0))      AX -= v[2*(idx-nx)+1];

if (idx < N)              d[idx] = AX;
*/

if(px<nx && py<ny)
{
float AX = 0;
if((px<(nx - 1))) AX += v[2 * (idx)+0];
if((px>0))      AX -= v[2 * (idx - 1) + 0];

if((py<(ny - 1)))
AX += v[2 * (idx)+1];
if((py>0))
AX -= v[2 * (idx - nx) + 1];

d[idx] = AX;
}
}