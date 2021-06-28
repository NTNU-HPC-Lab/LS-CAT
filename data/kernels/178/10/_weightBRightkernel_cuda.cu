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


__global__ void _weightBRightkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
{
int ind0, indg[3],q;
int k=blockIdx.x*blockDim.x + threadIdx.x;
if (k>=num_slices)
return;
ind0 = (num_grid-1) + (num_grid-1)*num_grid + k*num_grid*num_grid;

indg[0] = ind0-1;
indg[1] = ind0-num_grid;
indg[2] = ind0-num_grid-1;

for (q = 0; q < 3; q++) {
dev_F[ind0] += 2*beta*dev_wg3[q];
dev_G[ind0] -= 2*beta*dev_wg3[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
}
}