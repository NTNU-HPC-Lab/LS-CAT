#include "includes.h"
/*
* CCL3D.cu
*/


#define CCL_BLOCK_SIZE_X 8
#define CCL_BLOCK_SIZE_Y 8
#define CCL_BLOCK_SIZE_Z 8

__device__ int d_isNotDone;




__global__ void scanLabels(int* labels, int w, int h, int d) {
const int x = blockIdx.x * CCL_BLOCK_SIZE_X + threadIdx.x;
const int y = blockIdx.y * CCL_BLOCK_SIZE_Y + threadIdx.y;
const int z = blockIdx.z * CCL_BLOCK_SIZE_Z + threadIdx.z;
const int index = (z*h + y)*w + x;

if (x >= w || y >= h || z >= d) return;

const int Z1 = w*h; const int Y1 = w;

int lcur = labels[index];
if (lcur) {
int lmin = index; // MAX
// 26-neighbors
int lne, pos;
for (int Zdif = -Z1; Zdif <= Z1; Zdif += Z1) {
for (int Ydif = -Y1; Ydif <= Y1; Ydif += Y1) {
for (int Xdif = -1; Xdif <= 1; Xdif += 1) {
pos = index + Zdif + Ydif + Xdif;
lne = (pos >= 0 && pos < w*h*d) ? labels[pos] : 0; // circular boundary
if (lne && lne < lmin) lmin = lne;
}
}
}
// need not (Xdif,Ydif,Zdif)=(0,0,0) but no problem

if (lmin < lcur) {
int lpa = labels[lcur];
labels[lpa] = min(lpa, lmin);
d_isNotDone = 1;
}
}
}