#include "includes.h"
__device__ void Device_FloodFillZPlane(int zPlane, int L, int M, int N, unsigned char* vol)
{
long idx, idxS, idxN, ts;
bool anyChange = false;
int x, y;

ts = L*M*N;
// set point (0,0) to OUTSIZE_1
idx = zPlane*L*M /* + 0*L + 0 */;
vol[idx] = OUTSIDE_1;

anyChange = true;
while(anyChange) {

anyChange = false;
// loop from left to right and top to bottom
for(x=0; x < L; x++) {
for(y=0; y < M; y++) {
idxS = idx + y*L + x;
// if the point is set to OUTSIDE_1, the set all empty neightbors
// to OUTSIDE_1
if(vol[idxS] == OUTSIDE_1) {

idxN = idxS + L;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}

idxN = idxS - L;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}

idxN = idxS + 1;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}

idxN = idxS - 1;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}
}
}
}

if(anyChange) {
// same loop but bottom to top and right to left
anyChange = false;
// loop from left to right and top to bottom
for(x=L-1; x >=0; x--) {
for(y=M-1; y >=0; y--) {
idxS = idx + y*L + x;
// if the point is set to OUTSIDE_1, the set all empty neightbors
// to OUTSIDE_1
if(vol[idxS] == OUTSIDE_1) {

idxN = idxS + L;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}

idxN = idxS - L;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}

idxN = idxS + 1;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}

idxN = idxS - 1;
if((idxN >= 0) && (idxN < ts) && (vol[idxN] == 0)) {
vol[idxN] = OUTSIDE_1;
anyChange = true;
}
}
}
}
}
}


}
__global__ void kernel_for_z_planes(unsigned char *d_vol,int L,int M,int N)
{
Device_FloodFillZPlane(threadIdx.x,L,M,N,d_vol);
}