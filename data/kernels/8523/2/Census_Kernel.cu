#include "includes.h"
__global__ void Census_Kernel(unsigned char * MemSrc, unsigned int * MemDst, int eps, int Width, int Height)
{
//===============================================================================================
//
//===============================================================================================
int globalX = blockIdx.x * blockDim.x + threadIdx.x;
int globalY = blockIdx.y * blockDim.y + threadIdx.y;

int GlobalOffset = (globalY * Width + globalX);
float Value;
float ValueCenter;
unsigned int Census=0;
float Diff = 0;

//int threadX = threadIdx.x+3;
//int threadY = threadIdx.y+3;
//int blockDimX = blockDim.x+2*3;
//int blockDimY = blockDim.y+2*3;

//int OffsetLocal = (threadY * blockDimX + threadX);

extern __shared__ unsigned char DataCache[];
//FillCacheRadius(DataCache, MemSrc, 3, Width, Height);
//------------------------------------------------------------------
if (globalX>1 && globalX<(Width-2) && globalY>1 && globalY<(Height-2))
{
ValueCenter=MemSrc[GlobalOffset];
//ValueCenter=DataCache[OffsetLocal];

#pragma unroll
for(int dy=-1;dy<=1;dy++)
{
#pragma unroll
for(int dx=-1;dx<=1;dx++)
{
if (!(dx==0 && dy==0))
{
Value=MemSrc[(globalY+dy) * Width + (globalX+dx)];
//Value=DataCache[(threadY+dy) * blockDimX + (threadX+dx)];
//---------------------------------------------------------------------
// Ternary
//---------------------------------------------------------------------
Diff = ValueCenter - Value;

Census = Census << 2;

if (abs(Diff)<=eps)
{
Census=Census+1;
}
else if (Diff> eps)
{
Census=Census+2;
}
}
}
}
#pragma unroll
for(int dy=-2;dy<=2;dy++)
{
#pragma unroll
for(int dx=-2;dx<=2;dx++)
{
if (!(dx==0 && dy==0) && !(abs(dx)==1 || abs(dy)==1))
{
Value=MemSrc[(globalY+dy) * Width + (globalX+dx)];
//Value=DataCache[(threadY+dy) * blockDimX + (threadX+dx)];
//---------------------------------------------------------------------
// Ternary
//---------------------------------------------------------------------
Diff = ValueCenter - Value;
Census = Census << 2;

if (abs(Diff)<=eps)
{
Census=Census+1;
}
else if (Diff> eps)
{
Census=Census+2;
}
}
}
}
MemDst[GlobalOffset] = (Census);
}
else
{
if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
MemDst[GlobalOffset] = 0;
}
}