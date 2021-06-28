#include "includes.h"
/*
* Copyright: (C) 2017 VisLab, Institute for Systems and Robotics,
*                Instituto Superior Técnico, Universidade de Lisboa, Lisbon, Portugal
* Author: Pedro Vicente <pvicente@isr.tecnico.ulisboa.pt>
* CopyPolicy: Released under the terms of the GNU GPL v3.0.
*
*/



texture<float4, 2, cudaReadModeElementType> inTex;

// round up n/m
__global__ void CuDeviceArrayCopyFromTexture( float3* dst, int dstStep, int width, int height )
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if ( x > width || y > height ) return;

float4 res = tex2D(inTex, x, y);
float3* row_y = (float3*)((char*)dst + y * dstStep);
row_y[x] = make_float3(res.x, res.y, res.z);
}