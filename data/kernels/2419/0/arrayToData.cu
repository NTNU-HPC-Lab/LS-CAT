#include "includes.h"



//no performance difference if using float Mono input instead of float4 RGBA
//texture<float, cudaTextureType2D, cudaReadModeElementType> inTex;
//g_odata[offset] = tex2D(inTex, xc, yc);

texture<float4, cudaTextureType2D, cudaReadModeElementType> inTex;
surface<void, cudaSurfaceType2D> outputSurface;

__global__ void arrayToData(float *g_odata, uint* keys, int imgw, int imgh)
{

int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

int offset = x + y * imgw;

if (x < imgw && y < imgh) {

float xc = x + 0.5;
float yc = y + 0.5;


g_odata[offset] = tex2D(inTex, xc, yc).x;
keys[offset] = offset;
}


}