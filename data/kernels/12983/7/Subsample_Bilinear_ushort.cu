#include "includes.h"
__global__ void Subsample_Bilinear_ushort(cudaTextureObject_t ushort_tex, unsigned short *dst, int dst_width, int dst_height, int dst_pitch, int src_width, int src_height)
{
int xo = blockIdx.x * blockDim.x + threadIdx.x;
int yo = blockIdx.y * blockDim.y + threadIdx.y;

if (yo < dst_height && xo < dst_width)
{
float hscale = (float)src_width / (float)dst_width;
float vscale = (float)src_height / (float)dst_height;
float xi = (xo + 0.5f) * hscale;
float yi = (yo + 0.5f) * vscale;
// 3-tap filter weights are {wh,1.0,wh} and {wv,1.0,wv}
float wh = min(max(0.5f * (hscale - 1.0f), 0.0f), 1.0f);
float wv = min(max(0.5f * (vscale - 1.0f), 0.0f), 1.0f);
// Convert weights to two bilinear weights -> {wh,1.0,wh} -> {wh,0.5,0} + {0,0.5,wh}
float dx = wh / (0.5f + wh);
float dy = wv / (0.5f + wv);
int y0 = tex2D<unsigned short>(ushort_tex, xi-dx, yi-dy);
int y1 = tex2D<unsigned short>(ushort_tex, xi+dx, yi-dy);
int y2 = tex2D<unsigned short>(ushort_tex, xi-dx, yi+dy);
int y3 = tex2D<unsigned short>(ushort_tex, xi+dx, yi+dy);
dst[yo*dst_pitch+xo] = (unsigned short)((y0+y1+y2+y3+2) >> 2);
}
}