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


__global__ void _kernelpp_cuda(int num_projections, float mov, int num_pixels, int num_grid, int num_slices, float* dev_gridx, float* dev_gridy, float* dev_suma, float * dev_E, float* dev_data, float * dev_recon, float* dev_theta){
uint q = blockIdx.x*blockDim.x + threadIdx.x;
uint m = blockIdx.y*blockDim.y + threadIdx.y;
const double PI = 3.141592653589793238462;
bool quadrant;
float sinq, cosq;
float xi, yi;
float srcx, srcy, detx, dety;
float slope, islope;
int n,i,j,k;
int alen, blen, len;
int i1, i2;
float x1, x2;
int indx, indy;
int io;
float midx, midy, diffx, diffy;
float simdata;
float upd;
float coordx[MAX_NUM_GRID];
float coordy[MAX_NUM_GRID];
float ax[MAX_NUM_GRID];
float ay[MAX_NUM_GRID];
float bx[MAX_NUM_GRID];
float by[MAX_NUM_GRID];
float coorx[MAX_NUM_GRID*2];
float coory[MAX_NUM_GRID*2];
float leng[MAX_NUM_GRID*2];
int indi[MAX_NUM_GRID*2];
if((m>=num_pixels)||(q>=num_projections))
return;


// Calculate the sin and cos values
// of the projection angle and find
// at which quadrant on the cartesian grid.
sinq = sin(dev_theta[q]);
cosq =  cos(dev_theta[q]);
if ((dev_theta[q] >= 0 && dev_theta[q] < PI/2) ||
(dev_theta[q] >= PI && dev_theta[q] < 3*PI/2)) {
quadrant = true;
} else {
quadrant = false;
}

// Find the corresponding source and
// detector locations for a given line
// trajectory of a projection (Projection
// is specified by sinq and cosq).
xi = -1e6;
yi = -(num_pixels-1)/2.+m+mov;
srcx = xi*cosq-yi*sinq;
srcy = xi*sinq+yi*cosq;
detx = -xi*cosq-yi*sinq;
dety = -xi*sinq+yi*cosq;

// Find the intersection points of the
// line connecting the source and the detector
// points with the reconstruction grid. The
// intersection points are then defined as:
// (coordx, gridy) and (gridx, coordy)
slope = (srcy-dety)/(srcx-detx);
islope = 1/slope;

for (n = 0; n <= num_grid; n++) {
coordx[n] = islope*(dev_gridy[n]-srcy)+srcx;
coordy[n] = slope*(dev_gridx[n]-srcx)+srcy;
}

// Merge the (coordx, gridy) and (gridx, coordy)
// on a single array of points (ax, ay) and trim
// the coordinates that are outside the
// reconstruction grid.
alen = 0;
blen = 0;
for (n = 0; n <= num_grid; n++) {
if (coordx[n] > dev_gridx[0]) {
if (coordx[n] < dev_gridx[num_grid]) {
ax[alen] = coordx[n];
ay[alen] = dev_gridy[n];
alen++;
}
}
if (coordy[n] > dev_gridy[0]) {
if (coordy[n] < dev_gridy[num_grid]) {
bx[blen] = dev_gridx[n];
by[blen] = coordy[n];
blen++;
}
}
}
len = alen+blen;

// Sort the array of intersection points (ax, ay).
// The new sorted intersection points are
// stored in (coorx, coory).
i = 0;
j = 0;
k = 0;
if (quadrant) {
while (i < alen && j < blen)
{
if (ax[i] < bx[j]) {
coorx[k] = ax[i];
coory[k] = ay[i];
i++;
k++;
} else {
coorx[k] = bx[j];
coory[k] = by[j];
j++;
k++;
}
}
while (i < alen) {
coorx[k] = ax[i];
coory[k] = ay[i];
i++;
k++;
}
while (j < blen) {
coorx[k] = bx[j];
coory[k] = by[j];
j++;
k++;
}
} else {
while (i < alen && j < blen)
{
if (ax[alen-1-i] < bx[j]) {
coorx[k] = ax[alen-1-i];
coory[k] = ay[alen-1-i];
i++;
k++;
} else {
coorx[k] = bx[j];
coory[k] = by[j];
j++;
k++;
}
}

while (i < alen) {
coorx[k] = ax[alen-1-i];
coory[k] = ay[alen-1-i];
i++;
k++;
}
while (j < blen) {
coorx[k] = bx[j];
coory[k] = by[j];
j++;
k++;
}
}

// Calculate the distances (leng) between the
// intersection points (coorx, coory). Find
// the indices of the pixels on the
// reconstruction grid (indi).
for (n = 0; n < len-1; n++) {
diffx = coorx[n+1]-coorx[n];
diffy = coory[n+1]-coory[n];
leng[n] = sqrt(diffx*diffx+diffy*diffy);
midx = (coorx[n+1]+coorx[n])/2;
midy = (coory[n+1]+coory[n])/2;
x1 = midx+num_grid/2.;
x2 = midy+num_grid/2.;
i1 = (int)(midx+num_grid/2.);
i2 = (int)(midy+num_grid/2.);
indx = i1-(i1>x1);
indy = i2-(i2>x2);
indi[n] = indx+indy*num_grid;
}

// Note: The indices (indi) and the corresponding
// weights (leng) are the same for all slices. So,
// there is no need to calculate them for each slice.


//*******************************************************
// Below is for updating the reconstruction grid.

for (n = 0; n < len-1; n++) {
//        suma[indi[n]] += leng[n];
atomicAdd(&(dev_suma[indi[n]]),leng[n]);
}

for (k = 0; k < num_slices; k++) {
i = k*num_grid*num_grid;
io = m + k*num_pixels + q*num_slices*num_pixels;

simdata = 0;
for (n = 0; n < len-1; n++) {
simdata += dev_recon[indi[n]+i] * leng[n];
}
upd = dev_data[io]/simdata;
for (n = 0; n < len-1; n++) {
//            E[indi[n]+i] -= dev_recon[indi[n]+i]*upd*leng[n];
atomicAdd(&(dev_E[indi[n]+i]),-dev_recon[indi[n]+i]*upd*leng[n]);
}
}
}