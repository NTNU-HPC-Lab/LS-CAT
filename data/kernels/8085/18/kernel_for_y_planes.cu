#include "includes.h"
__device__ void Device_FloodFillYPlane(int yPlane, int L, int M, int N, unsigned char* vol)
{
long idx, idxS, idxN, ts;
bool anyChange = false;
int x, z;

ts = L*M*N;
// set point (0,0) to OUTSIZE_2
idx = /*0*L*M  + */ yPlane*L /*+ 0 */;
vol[idx] = OUTSIDE_2;

anyChange = true;
while(anyChange) {

anyChange = false;
// loop from left to right and top to bottom
for(x=0; x < L; x++) {
for(z=0; z < N; z++) {
idxS = z*L*M + idx + x;
// if the point is set to OUTSIDE_2, the set all empty neightbors
// to OUTSIDE_2
if(vol[idxS] == OUTSIDE_2) {

idxN = idxS + L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}

idxN = idxS - L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}

idxN = idxS + 1;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}

idxN = idxS - 1;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}
}
}
}

if(anyChange) {
// same loop but bottom to top and right to left

anyChange = false;
// loop from left to right and top to bottom
for(x=L-1; x >= 0; x--) {
for(z=N-1; z >= 0; z--) {
idxS = z*L*M + idx + x;
// if the point is set to OUTSIDE_2, the set all empty neightbors
// to OUTSIDE_2
if(vol[idxS] == OUTSIDE_2) {

idxN = idxS + L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}

idxN = idxS - L*M;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}

idxN = idxS + 1;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}

idxN = idxS - 1;
if((idxN >= 0) && (idxN < ts) &&
((vol[idxN] == 0) || (vol[idxN] == OUTSIDE_1))) {
vol[idxN] = OUTSIDE_2;
anyChange = true;
}
}
}
}
}
}



}
__global__ void kernel_for_y_planes(unsigned char *d_vol,int L,int M,int N)
{
Device_FloodFillYPlane(threadIdx.x,L,M,N,d_vol);
}