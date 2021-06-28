#include "includes.h"



//no performance difference if using float Mono input instead of float4 RGBA
//texture<float, cudaTextureType2D, cudaReadModeElementType> inTex;
//g_odata[offset] = tex2D(inTex, xc, yc);

texture<float4, cudaTextureType2D, cudaReadModeElementType> inTex;
surface<void, cudaSurfaceType2D> outputSurface;

__global__ void dataToTex(uint* indices, float4 *g_odata, int imgw, int imgh)
{

int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

int offset = x + y * imgw;

if (x < imgw && y < imgh) {

float res = indices[offset];
g_odata[offset] = make_float4(res, 0, 0, 1);
}

}