#include "includes.h"
__global__ void addForces_k(float2 *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch) {

int tx = threadIdx.x;
int ty = threadIdx.y;
float2 *fj = (float2*)((char*)v + (ty + spy) * pitch) + tx + spx;

float2 vterm = *fj;
tx -= r; ty -= r;
float s = 1.f / (1.f + tx*tx*tx*tx + ty*ty*ty*ty);
vterm.x += s * fx;
vterm.y += s * fy;
*fj = vterm;
}