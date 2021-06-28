#include "includes.h"
/// Copyright (C) 2016 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>
/// License: GPLv3


#define restrict __restrict__

typedef unsigned int uint;
typedef unsigned int hashKey;
typedef ushort4 particleinfo;

__global__ void initParticles( particleinfo * restrict infoArray, hashKey * restrict hashArray, uint * restrict idxArray, uint numParticles)
{
uint idx = threadIdx.x + blockIdx.x*blockDim.x;

if (idx > numParticles)
return;

idxArray[idx] = idx;

particleinfo info;
info.x = idx % 4;
info.y = 0;
info.z = (ushort)(idx & 0xffff);
info.w = (ushort)(idx >> 16);

infoArray[idx] = info;

hashArray[idx] = idx/17 + (idx % (idx & 17));
}