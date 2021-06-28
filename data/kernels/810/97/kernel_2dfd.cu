#include "includes.h"
__global__ void kernel_2dfd(float *g_u1, float *g_u2, const int nx, const int iStart, const int iEnd)
{
// global to line index
unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;

// smem idx for current point
unsigned int stx = threadIdx.x + NPAD;
unsigned int idx  = ix + iStart * nx;

// shared memory for x dimension
__shared__ float line[BDIMX + NPAD2];

// a coefficient related to physical properties
const float alpha = 0.12f;

// register for y value
float yval[9];

for (int i = 0; i < 8; i++) yval[i] = g_u2[idx + (i - 4) * nx];

// skip for the bottom most y value
int iskip = NPAD * nx;

#pragma unroll 9
for (int iy = iStart; iy < iEnd; iy++)
{
// get yval[8] here
yval[8] = g_u2[idx + iskip];

// read halo part
if(threadIdx.x < NPAD)
{
line[threadIdx.x]  = g_u2[idx - NPAD];
line[stx + BDIMX]    = g_u2[idx + BDIMX];
}

line[stx] = yval[4];
__syncthreads();

// 8rd fd operator
if ( (ix >= NPAD) && (ix < nx - NPAD) )
{
// center point
float tmp = coef[0] * line[stx] * 2.0f;

#pragma unroll
for(int d = 1; d <= 4; d++)
{
tmp += coef[d] * ( line[stx - d] + line[stx + d]);
}

#pragma unroll
for(int d = 1; d <= 4; d++)
{
tmp += coef[d] * (yval[4 - d] + yval[4 + d]);
}

// time dimension
g_u1[idx] = yval[4] + yval[4] - g_u1[idx] + alpha * tmp;
}

#pragma unroll 8
for (int i = 0; i < 8 ; i++)
{
yval[i] = yval[i + 1];
}

// advancd on global idx
idx  += nx;
__syncthreads();
}
}