#include "includes.h"

#define KERNEL_RADIUS 31
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c_Kernel[ KERNEL_LENGTH ];

__global__ void convolutionY_63_Kernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, int outofbounds, float outofboundsvalue )
{
__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

//Offset to the upper halo edge
const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z;

const int firstPixelInLine = (COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS - threadIdx.y) * imageW;
const int lastPixelInLine = (imageH - baseY - 1) * imageW;

d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

//Main data
#pragma unroll

for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : outofboundsvalue;
else
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : d_Src[ lastPixelInLine ];
}

//Upper halo
#pragma unroll

for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : outofboundsvalue;
else
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : d_Src[ firstPixelInLine ];
}

//Lower halo
#pragma unroll

for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : outofboundsvalue;
else
s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : d_Src[ lastPixelInLine ];
}

//Compute and store results
__syncthreads();

// this pixel is not part of the image and does not need to be convolved
if ( baseX >= imageW )
return;

#pragma unroll

for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
{
if (imageH - baseY > i * COLUMNS_BLOCKDIM_Y)
{
float sum = 0;

#pragma unroll

for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
{
sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
}

d_Dst[i * COLUMNS_BLOCKDIM_Y * imageW] = sum;
}
}
}