#include "includes.h"

#define KERNEL_RADIUS 31
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c_Kernel[ KERNEL_LENGTH ];

__global__ void convolutionX_63_Kernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, int outofbounds, float outofboundsvalue )
{
__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z;

const int firstPixelInLine = ROWS_BLOCKDIM_X * ROWS_HALO_STEPS - threadIdx.x;
const int lastPixelInLine = imageW - baseX - 1;

// set the input and output arrays to the right offset (actually the output is not at the right offset, but this is corrected later)
d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

// Load main data
// Start copying after the ROWS_HALO_STEPS, only the original data that will be convolved
#pragma unroll

for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : outofboundsvalue;
else
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : d_Src[ lastPixelInLine ];
}

// Load left halo
// If the data fetched is outside of the image (note: baseX can be <0 for the first block) , use a zero-out of bounds strategy
#pragma unroll

for (int i = 0; i < ROWS_HALO_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : outofboundsvalue;
else
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : d_Src[ firstPixelInLine ];
}

//Load right halo
#pragma unroll

for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : outofboundsvalue;
else
s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : d_Src[ lastPixelInLine ];
}

//Compute and store results
__syncthreads();

// this pixel is not part of the image and does not need to be convolved
if ( baseY >= imageH )
return;

#pragma unroll

for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
{
if (imageW - baseX > i * ROWS_BLOCKDIM_X)
{
float sum = 0;

#pragma unroll

for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
{
sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
}

d_Dst[i * ROWS_BLOCKDIM_X] = sum;
}
}
}