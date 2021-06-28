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

__global__ void conjugateComplexMulKernel(const float2* __restrict__ aIn, float2* __restrict__ bInOut, int maxElem)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= maxElem)
return;

float2 valA = aIn[idx];
//conjugate complex
valA.y = -valA.y;

//multiplication:
float2 valB = bInOut[idx];
float2 res;
res.x = valA.x * valB.x - valA.y * valB.y;
res.y = valA.x * valB.y + valA.y * valB.x;
bInOut[idx] = res;
}