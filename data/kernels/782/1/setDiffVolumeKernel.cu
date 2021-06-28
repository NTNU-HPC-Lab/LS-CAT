#include "includes.h"
__global__ void setDiffVolumeKernel(float *d_fv, unsigned char *d_picture1, unsigned char *d_picture2, unsigned picWidth, unsigned picHeight) {
__shared__ float p1_section[10 * 10 * 4];
__shared__ float p2_section[10 * 10 * 4];
unsigned i;

// This thread's position in its block's subsection of the float volume
unsigned sx, sy, sz;
// Dimensions of the grid
unsigned gx, gy, gz;
// Position of this thread's block
unsigned bx, by, bz;
// This thread's position in the entire float volume
unsigned vx, vy, vz;
// The location of the colors that this thread will be comparing
unsigned c1, c2;

// Get the position of this thread in its subsection
sz = threadIdx.x % 10;
sy = threadIdx.x / 100;
sx = (threadIdx.x % 100) / 10;

// Get the dimensions of the grid
gz = picWidth / 10;
if(picWidth % 10) gz++;
gy = picHeight / 10;
if(picHeight % 10) gy++;
gx = picWidth / 10;
if(picWidth % 10) gx++;

// Get the position of this thread's block
bz = blockIdx.x % gz;
by = blockIdx.x / (gx * gz);
bx = (blockIdx.x % (gx * gz)) / gz;

// Get the position of this thread in entire float volume
vx = sx + 10 * bx;
vy = sy + 10 * by;
vz = sz + 10 * bz;

// Copy subpicture to shared memory

// See if this thread needs to copy from picture 1
// picture 1 covers width * height

// If the float volume z of this thread is zero,
// then it needs to copy from picture 1
if(sz == 0) {
// Check if this thread will get a pixel not in the picture
if(vx < picWidth && vy < picHeight) {
for(i = 0; i < 4; i++) {
p1_section[(sx + sy * 10) * 4 + i] =
(float) d_picture1[(vx + vy * picWidth) * 4 + i];
}
}
}

// See if this thread needs to copy from picture 2
// picture 2 covers depth * height

// If the float volume x of this thread is zero,
// then it needs to copy from picture 2
if(sx == 0) {
// Check if this thread will get a pixel not in the picture
if(vz < picWidth && vy < picHeight) {
for(i = 0; i < 4; i++) {
p2_section[(sz + sy * 10) * 4 + i] =
(float) d_picture2[(vz + vy * picWidth) * 4 + i];
}
}
}

__syncthreads();
// Now each of d_picture1 and d_picture2 are properly filled out

// Write difference into float volume
if(vx < picWidth && vy < picHeight && vz < picWidth) {
c1 = (sx + sy * 10) * 4;
c2 = (sz + sy * 10) * 4;
d_fv[vz + vx * picWidth + vy * picWidth * picWidth] =
sqrtf(
powf(p1_section[c1 + 0] - p2_section[c2 + 0], 2.f) +
powf(p1_section[c1 + 1] - p2_section[c2 + 1], 2.f) +
powf(p1_section[c1 + 2] - p2_section[c2 + 2], 2.f) +
powf(p1_section[c1 + 3] - p2_section[c2 + 3], 2.f)
);
}
}