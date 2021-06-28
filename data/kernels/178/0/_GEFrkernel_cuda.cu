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


__global__ void _GEFrkernel_cuda(int num_slices,int num_grid,float* dev_recon,float* dev_G,float* dev_E,float* dev_F)
{
uint m = blockIdx.x*blockDim.x + threadIdx.x;
uint n = blockIdx.y*blockDim.y + threadIdx.y;
uint k = blockIdx.z;
uint i = m + n*num_grid + k*num_grid*num_grid;
if((m>=num_grid)||(n>=num_grid)||(k>=num_slices))
return;
//	int i = m + n*num_grid + k*num_grid*num_grid;
//  recon[i] = (-G[i]+sqrt(G[i]*G[i]-8*E[i]*F[i]))/(4*F[i]);
dev_recon[i] = (-dev_G[i]+sqrtf(dev_G[i]*dev_G[i]-8.*dev_E[i]*dev_F[i]))/(4.*dev_F[i]);
}