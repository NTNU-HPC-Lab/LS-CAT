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


__global__ void _weightLeftkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
{
uint q;
int ind0, indg[5];
uint k = blockIdx.x*blockDim.x + threadIdx.x;
uint n = blockIdx.y*blockDim.y + threadIdx.y+1;

if ((k>=num_slices)||(n<1)||(n>=(num_grid-1)))
return;

ind0 = n*num_grid + k*num_grid*num_grid;

indg[0] = ind0+1;
indg[1] = ind0+num_grid;
indg[2] = ind0-num_grid;
indg[3] = ind0+num_grid+1;
indg[4] = ind0-num_grid+1;

for (q = 0; q < 5; q++) {
dev_F[ind0] += 2*beta*dev_wg5[q];
dev_G[ind0] -= 2*beta*dev_wg5[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
}
}