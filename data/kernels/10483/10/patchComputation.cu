#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void patchComputation(int noCandidates, int W, int H, int skpx, int skpy, int xres, int yres, float subPatchArea, float xspacing, float yspacing, float capacity, int uniqueRegions, const int* labelledImage, const float* pops, float* results) {

// Get global index of thread
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < noCandidates) {
// Dimensions arranged as X->Y->R
int rem = idx;
int blockIdxY = (int)(idx/(xres*uniqueRegions));
rem = rem - blockIdxY*(xres*uniqueRegions);
int blockIdxX = (int)(rem/uniqueRegions);
rem = rem - blockIdxX*(uniqueRegions);
// Valid region numbering starts at 1, not 0
int regionNo = rem + 1;

int blockSizeX;
int blockSizeY;

if ((blockIdxX+1)*skpx <= H) {
blockSizeX = skpx;
} else {
blockSizeX = H-blockIdxX*skpx;
}

if ((blockIdxY+1)*skpy <= W) {
blockSizeY = skpy;
} else {
blockSizeY = W-blockIdxY*skpy;
}

// Iterate through each sub patch for this large grid cell
float area = 0.0f;
float cap = 0.0f;
float pop = 0.0f;
float cx = 0.0f;
float cy = 0.0f;

for (int ii = 0; ii < blockSizeX; ii++) {
for (int jj = 0; jj < blockSizeY; jj++) {
int xCoord = blockIdxX*skpx+ii;
int yCoord = blockIdxY*skpy+jj;

area += (float)(labelledImage[xCoord + yCoord*W] == regionNo);
}
}

if (area > 0) {
for (int ii = 0; ii < blockSizeX; ii++) {
for (int jj = 0; jj < blockSizeY; jj++) {
int xCoord = blockIdxX*skpx+ii;
int yCoord = blockIdxY*skpy+jj;

if (labelledImage[xCoord + yCoord*W] == regionNo) {
pop += (float)pops[xCoord + yCoord*W];
cx += ii;
cy += jj;
}
}
}
cx = xspacing*(cx/area + blockIdxX*skpx);
cy = yspacing*(cy/area + blockIdxY*skpy);
area = area*subPatchArea;
cap = area*capacity;
}

// Store results to output matrix
results[5*idx] = area;
results[5*idx+1] = cap;
results[5*idx+2] = pop;
results[5*idx+3] = cx;
results[5*idx+4] = cy;
}
}