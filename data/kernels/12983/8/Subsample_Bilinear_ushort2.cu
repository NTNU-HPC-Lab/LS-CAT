#include "includes.h"
__global__ void Subsample_Bilinear_ushort2(cudaTextureObject_t ushort2_tex, ushort2 *dst, int dst_width, int dst_height, int dst_pitch2, int src_width, int src_height)
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
ushort2 c0 = tex2D<ushort2>(ushort2_tex, xi-dx, yi-dy);
ushort2 c1 = tex2D<ushort2>(ushort2_tex, xi+dx, yi-dy);
ushort2 c2 = tex2D<ushort2>(ushort2_tex, xi-dx, yi+dy);
ushort2 c3 = tex2D<ushort2>(ushort2_tex, xi+dx, yi+dy);
int2 uv;
uv.x = ((int)c0.x+(int)c1.x+(int)c2.x+(int)c3.x+2) >> 2;
uv.y = ((int)c0.y+(int)c1.y+(int)c2.y+(int)c3.y+2) >> 2;
dst[yo*dst_pitch2+xo] = make_ushort2((unsigned short)uv.x, (unsigned short)uv.y);
}
}