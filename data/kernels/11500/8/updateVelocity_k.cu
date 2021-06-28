#include "includes.h"
__global__ void updateVelocity_k(float2 *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch) {

int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
int p;

float vxterm, vyterm;
float2 nvterm;
// gtidx is the domain location in x for this thread
if (gtidx < dx) {
for (p = 0; p < lb; p++) {
// fi is the domain location in y for this thread
int fi = gtidy + p;
if (fi < dy) {
int fjr = fi * pdx + gtidx;
vxterm = vx[fjr];
vyterm = vy[fjr];

// Normalize the result of the inverse FFT
float scale = 1.f / (dx * dy);
nvterm.x = vxterm * scale;
nvterm.y = vyterm * scale;

float2 *fj = (float2*)((char*)v + fi * pitch) + gtidx;
*fj = nvterm;
}
} // If this thread is inside the domain in Y
} // If this thread is inside the domain in X
}