#include "includes.h"
//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//


///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

__constant__ float d_Threshold[2];
__constant__ float d_Scales[8], d_Factor;
__constant__ float d_EdgeLimit;
__constant__ int d_MaxNumPoints;

__device__ unsigned int d_PointCounter[1];
__constant__ float d_Kernel1[5];
__constant__ float d_Kernel2[12*16];

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter an subsample image
///////////////////////////////////////////////////////////////////////////////
__global__ void ScaleDown_D(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch) {
// TODO: one element per thread in a block?
__shared__ float inrow[SCALEDOWN_W + 4];

__shared__ float brow[5 * (SCALEDOWN_W / 2)];

//
__shared__ int yRead[SCALEDOWN_H + 4];
__shared__ int yWrite[SCALEDOWN_H + 4];

// Get thread index, which ranges from 0 to SCALEDOWN_W + 4
const int tx = threadIdx.x;

// Get indices in brow
// TODO: move this out?
#define dx2 (SCALEDOWN_W / 2)
const int tx0 = tx + 0 * dx2;
const int tx1 = tx + 1 * dx2;
const int tx2 = tx + 2 * dx2;
const int tx3 = tx + 3 * dx2;
const int tx4 = tx + 4 * dx2;

// TODO: x and y pixel index
const int xStart = blockIdx.x * SCALEDOWN_W;
const int yStart = blockIdx.y * SCALEDOWN_H;

// TODO: x coordinate to write to?
const int xWrite = xStart / 2 + tx;
int xRead = xStart + tx - 2;
xRead = (xRead < 0 ? 0 : xRead);
xRead = (xRead >= width ? width - 1 : xRead);

const float *k = d_Kernel1;

// Identify y read and write indices; note we ignore SCALEDOWN_H + 4 <= tx <
// SCALEDOWN_H + 4 in this section
if (tx < SCALEDOWN_H + 4) {
// TODO: tx = 0 and tx = 1 are the same; why?
int y = yStart + tx - 1;

// Clamp at 0 and height - 1
y = (y < 0 ? 0 : y);
y = (y >= height ? height - 1 : y);

// Read start index
yRead[tx] = y * pitch;

// Write start index
yWrite[tx] = (yStart + tx - 4) / 2 * newpitch;
}

// Synchronize threads to ensure we have yRead and yWrite filled for current
// warp
__syncthreads();

// For each thread (which runs 0 to SCALEDOWN_W + 4 - 1), loop through 0 to
// SCALEDOWN_H + 4 - 1 by kernel size.
for (int dy = 0; dy < SCALEDOWN_H + 4; dy += 5) {

// yRead[dy + 0] is the y index to 0th row of data from source image (may
// be the same as 1st, 2nd, etc row, depending on how close we are to the
// edge of image). xRead is determined by thread id and starts from size
// of kernel / 2 + 1 to the left of our current pixel
inrow[tx] = d_Data[yRead[dy + 0] + xRead];

// Once we synchronize, inrow should contain the data from the source
// image corresponding to the first row in the current block. It is length
// SCALEDOWN_W + 4.
__syncthreads();

// For the SCALEDOWN_W / 2 threads in block, compute the first of 5
// indices for this thread. Convolve the 1-D kernel k with every other
// 'pixel' in the block via 2 * tx
if (tx < dx2) {
brow[tx0] = k[0] * (inrow[2 * tx] + inrow[2 * tx + 4]) +
k[1] * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) +
k[2] * inrow[2 * tx + 2];
}

// TODO: Once we synchronize, brow[tx0] should contain
__syncthreads();

// Compute for SCALEDOWN_W / 2 threads in block. dy & 1 is true if dy is
// odd. We require that dy is even and after we've completed at least one
// iteration
if (tx < dx2 && dy >= 4 && !(dy & 1)) {
d_Result[yWrite[dy + 0] + xWrite] = k[2] * brow[tx2] +
k[0] * (brow[tx0] + brow[tx4]) +
k[1] * (brow[tx1] + brow[tx3]);
}

// And...this is all just the same as above. One big unrolled for loop.
if (dy < (SCALEDOWN_H + 3)) {
// yRead[dy + 1] is the y index to 1th row of data from source image
// (may be the same as 1st, 2nd, etc row, depending on how close we are
// to the edge of image). xRead is determined by thread id and starts
// from size of kernel / 2 + 1 to the left of our current pixel
inrow[tx] = d_Data[yRead[dy + 1] + xRead];

__syncthreads();
if (tx < dx2) {
brow[tx1] = k[0] * (inrow[2 * tx] + inrow[2 * tx + 4]) +
k[1] * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) +
k[2] * inrow[2 * tx + 2];
}
__syncthreads();
if (tx<dx2 && dy>=3 && (dy&1)) {
d_Result[yWrite[dy+1] + xWrite] = k[2]*brow[tx3] + k[0]*(brow[tx1]+brow[tx0]) + k[1]*(brow[tx2]+brow[tx4]);
}
}
if (dy<(SCALEDOWN_H+2)) {
inrow[tx] = d_Data[yRead[dy+2] + xRead];
__syncthreads();
if (tx<dx2) {
brow[tx2] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
}
__syncthreads();
if (tx<dx2 && dy>=2 && !(dy&1)) {
d_Result[yWrite[dy+2] + xWrite] = k[2]*brow[tx4] + k[0]*(brow[tx2]+brow[tx1]) + k[1]*(brow[tx3]+brow[tx0]);
}
}
if (dy<(SCALEDOWN_H+1)) {
inrow[tx] = d_Data[yRead[dy+3] + xRead];
__syncthreads();
if (tx<dx2) {
brow[tx3] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
}
__syncthreads();
if (tx<dx2 && dy>=1 && (dy&1)) {
d_Result[yWrite[dy+3] + xWrite] = k[2]*brow[tx0] + k[0]*(brow[tx3]+brow[tx2]) + k[1]*(brow[tx4]+brow[tx1]);
}
}
if (dy<SCALEDOWN_H) {
inrow[tx] = d_Data[yRead[dy+4] + xRead];
__syncthreads();
if (tx<dx2) {
brow[tx4] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
}
__syncthreads();
if (tx<dx2 && !(dy&1)) {
d_Result[yWrite[dy+4] + xWrite] = k[2]*brow[tx1] + k[0]*(brow[tx4]+brow[tx3]) + k[1]*(brow[tx0]+brow[tx2]);
}
}
__syncthreads();
}
}