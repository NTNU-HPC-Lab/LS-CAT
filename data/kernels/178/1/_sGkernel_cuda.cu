#include "includes.h"
/*
Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com

This file is part of TomograPeri.

TomograPeri is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TomograPeri is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.
*/



#define blockx 16
#define blocky 16


__global__ void _sGkernel_cuda(int num_slices, int num_grid, float* dev_G,float *dev_suma)
{
uint m = blockIdx.x*blockDim.x + threadIdx.x;
uint n = blockIdx.y*blockDim.y + threadIdx.y;
uint k = blockIdx.z;
uint i = m + n*num_grid + k*num_grid*num_grid;
uint j = m + n*num_grid;
if((m>=num_grid)||(n>=num_grid)||(k>=num_slices))
return;
//	G[k*num_grid*num_grid+n] += suma[n];
dev_G[i]+= dev_suma[j];
}