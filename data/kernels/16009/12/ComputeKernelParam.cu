#include "includes.h"
//	Copyright (c) 2020, Michael Kunz. All rights reserved.
//	https://github.com/kunzmi/ImageStackAlignator
//
//	This file is part of ImageStackAlignator.
//
//	ImageStackAlignator is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as
//	published by the Free Software Foundation, version 3.
//
//	ImageStackAlignator is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.




//squared sum of a tile without the border
extern "C"

//Boxfilter ignoring the border parts
//blockDim.X must be tileSize + 2 * maxShift
//blockDim.Y must be 1
extern "C"
//Boxfilter ignoring the border parts
//blockDim.Y must be tileSize + 2 * maxShift
//blockDim.X must be 1
extern "C"


//Computed the normalized CC values out of the different input data
//Cross correlation is fft shifted
//blockDim.X must be 2 * maxShift
//blockDim.Y must be 2 * maxShift
//blockDim.Z must be nr of tiles
extern "C"

//Convert a tiled image into consecutive tiles for FFT
//input img has a pitch, output tiles are consecutive
//output tiles overlap by maxShift is filled by zero
extern "C"

//Convert a tiled image into consecutive tiles for FFT
//input img has a pitch, output tiles are consecutive
//output tiles overlap by maxShift on each side
extern "C"

__global__ void ComputeKernelParam( float3* __restrict__ kernelImg, int imgWidth, int imgHeight, int imgOutPitch, float Dth, float Dtr, float kDetail, float kDenoise, float kStretch, float kShrink)
{
int pxX = blockIdx.x * blockDim.x + threadIdx.x;
int pxY = blockIdx.y * blockDim.y + threadIdx.y;

if (pxX >= imgWidth || pxY >= imgHeight)
return;

float3 grad = *(((float3*)((char*)kernelImg + imgOutPitch * pxY)) + pxX);
float a11 = grad.x;
float a22 = grad.y;
float a12 = grad.z;

float help = sqrtf((a22 - a11) * (a22 - a11) + 4.0f * a12 * a12);
float c = 2.0f * a12;
float s = a22 - a11 + help;

float norm = sqrtf(c * c + s * s);
if (norm > 0)
{
c /= norm;
s /= norm;
}
else
{
c = 1;
s = 0;
}

float lam1 = (a11 + a22 + help) / 2.0f;
float lam2 = (a11 + a22 - help) / 2.0f;


float A = 1 + sqrtf((lam1 - lam2) * (lam1 - lam2) / ((lam1 + lam2) * (lam1 + lam2)));
float D = 1 - sqrtf(lam1) / Dtr + Dth;

D = fmaxf(fminf(1.0f, D), 0.0f);

float k1h = kDetail * kStretch * A;
float k2h = kDetail / kShrink * A;

float k1 = ((1.0f - D)*k1h + D*kDetail*kDenoise);
float k2 = ((1.0f - D)*k2h + D*kDetail*kDenoise);
k1 *= k1;
k2 *= k2;

float x2 = c;
float y2 = s;
float x1 = s;
float y1 = -c;

float b11 = k1*x1*x1 + x2*x2*k2;
float b12 = k1*x1*y1 + x2*y2*k2;
float b22 = k1*y1*y1 + y2*y2*k2;

float det = b11*b22 - b12*b12 + 0.0000000001f;

float3 kernel;
kernel.x = b22 / det;
kernel.y = b11 / det;
kernel.z = -b12 / det;
*(((float3*)((char*)kernelImg + imgOutPitch * pxY)) + pxX) = kernel;
}