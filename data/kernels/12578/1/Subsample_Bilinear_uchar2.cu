#include "includes.h"
/*
* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

extern "C" {

texture<unsigned char, 2> uchar_tex;
texture<uchar2, 2>  uchar2_tex;
texture<uchar4, 2>  uchar4_tex;
texture<unsigned short, 2> ushort_tex;
texture<ushort2, 2>  ushort2_tex;
texture<ushort4, 2>  ushort4_tex;







}
__global__ void Subsample_Bilinear_uchar2(uchar2 *dst, int dst_width, int dst_height, int dst_pitch2, int src_width, int src_height)
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
uchar2 c0 = tex2D(uchar2_tex, xi-dx, yi-dy);
uchar2 c1 = tex2D(uchar2_tex, xi+dx, yi-dy);
uchar2 c2 = tex2D(uchar2_tex, xi-dx, yi+dy);
uchar2 c3 = tex2D(uchar2_tex, xi+dx, yi+dy);
int2 uv;
uv.x = ((int)c0.x+(int)c1.x+(int)c2.x+(int)c3.x+2) >> 2;
uv.y = ((int)c0.y+(int)c1.y+(int)c2.y+(int)c3.y+2) >> 2;
dst[yo*dst_pitch2+xo] = make_uchar2((unsigned char)uv.x, (unsigned char)uv.y);
}
}