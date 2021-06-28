#include "includes.h"

#define KERNEL_RADIUS 31
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c_Kernel[ KERNEL_LENGTH ];

__global__ void convolutionZ_63_Kernel( float *d_Dst, float *d_Src, int imageW, int imageH, int imageD, int outofbounds, float outofboundsvalue )
{
// here it is [x][z], we leave out y as it has a size of 1
__shared__ float s_Data[DEPTH_BLOCKDIM_X][(DEPTH_RESULT_STEPS + 2 * DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + 1];

//Offset to the upper halo edge
const int baseX = blockIdx.x * DEPTH_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y;
const int baseZ = (blockIdx.z * DEPTH_RESULT_STEPS - DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + threadIdx.z;

const int firstPixelInLine = (DEPTH_BLOCKDIM_Z * DEPTH_HALO_STEPS - threadIdx.z) * imageW * imageH;
const int lastPixelInLine = (imageD - baseZ - 1) * imageW * imageH;

d_Src += baseZ * imageH * imageW + baseY * imageW + baseX;
d_Dst += baseZ * imageH * imageW + baseY * imageW + baseX;

//Main data
#pragma unroll

for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : outofboundsvalue;
else
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : d_Src[ lastPixelInLine ];
}

//Upper halo
#pragma unroll

for (int i = 0; i < DEPTH_HALO_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (baseZ >= -i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (baseZ >= -i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : outofboundsvalue;
else
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (baseZ >= -i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : d_Src[ firstPixelInLine ];
}

//Lower halo
#pragma unroll

for (int i = DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS + DEPTH_HALO_STEPS; i++)
{
if ( outofbounds == 0 )
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : 0;
else if ( outofbounds == 1 )
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : outofboundsvalue;
else
s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * imageW * imageH] : d_Src[ lastPixelInLine ];
}

//Compute and store results
__syncthreads();

// this pixel is not part of the image and does not need to be convolved
if ( baseX >= imageW )
return;

#pragma unroll

for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
{
if (imageD - baseZ > i * DEPTH_BLOCKDIM_Z)
{
float sum = 0;

#pragma unroll

for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
{
sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z + j];
}

d_Dst[i * DEPTH_BLOCKDIM_Z * imageW * imageH] = sum;
}
}
}