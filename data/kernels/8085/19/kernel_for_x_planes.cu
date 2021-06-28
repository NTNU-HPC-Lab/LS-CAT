#include "includes.h"
__device__ void Device_FloodFillXPlane(int xPlane, int L, int M, int N, unsigned char* vol)
{

long idx, idxS, idxN, ts;
bool anyChange = false;
int y, z;

ts = L*M*N;
// set point (0,0) to OUTSIZE_3
idx = /*0*L*M  +  yPlane*L */+ xPlane ;
vol[idx] = OUTSIDE_3;

anyChange = true;
while(anyChange) {

anyChange = false;
// loop from left to right and top to bottom
for(y=0; y < M; y++) {
for(z=0; z < N; z++) {
idxS = z*L*M + L*y + idx;
// if the point is set to OUTSIDE_3, the set all empty neightbors
// to OUTSIDE_3
if(vol[idxS] == OUTSIDE_3) {

idxN = idxS + L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1) || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}

idxN = idxS - L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1) || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}

idxN = idxS + L;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1) || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}

idxN = idxS - L;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1) || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}
}
}
}

if(anyChange) {
// same loop but bottom to top and right to left

anyChange = false;
// loop from left to right and top to bottom
for(y=M-1; y >= 0; y--) {
for(z=N-1; z >= 0; z--) {
idxS = z*L*M + + L*y + idx;
// if the point is set to OUTSIDE_3, the set all empty neightbors
// to OUTSIDE_3
if(vol[idxS] == OUTSIDE_3) {

idxN = idxS + L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1)  || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}

idxN = idxS - L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1) || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}

idxN = idxS + L;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1) || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}

idxN = idxS - L;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1) || (vol[idxN] == OUTSIDE_2))) {
vol[idxN] = OUTSIDE_3;
anyChange = true;
}
}
}
}
}
}



}
__global__ void kernel_for_x_planes(unsigned char *d_vol,int L,int M,int N)
{
Device_FloodFillXPlane(threadIdx.x,L,M,N,d_vol);
}