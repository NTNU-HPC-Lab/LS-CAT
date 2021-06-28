#include "includes.h"


typedef unsigned int  uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;
texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex_block;
texture<float4,  1, cudaReadModeElementType> texture_float_1D;

struct Ray
{
float3 o;   // origin
float3 d;   // direction
};

__device__
__device__ unsigned char myMAX(unsigned char a, unsigned char b)
{
if(a >= b)
return a;
else
return b;
}
__global__ void block_volume(unsigned char* image_p, unsigned char* dest_p, int srcWidth, int srcHeight, int srcDepth, int desWidth, int desHeight, int desDepth){


unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;

if (tx >= desWidth || ty >= desHeight) return;

for(int i=0; i<desDepth; i++){
dest_p[i*desWidth*desHeight + ty*desHeight + tx] = 0;
unsigned char tempmax=0;

for(int z=i*4; z<=i*4+4; z++)
for(int y=ty*4; y<=ty*4+4; y++)
for(int x=tx*4; x<=tx*4+4; x++){
if(z>=srcDepth || y>=srcHeight || x>=srcWidth )
continue;
tempmax = myMAX(tempmax, image_p[z*srcWidth*srcHeight + y*srcHeight + x]);
}
dest_p[i*desWidth*desHeight + ty*desHeight + tx] = tempmax;
}

}