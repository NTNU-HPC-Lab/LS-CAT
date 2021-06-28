#include "includes.h"
__global__ void kInitClusters(const cudaSurfaceObject_t surfFrameLab, float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol) {
int centroidIdx = blockIdx.x*blockDim.x + threadIdx.x;
int nSpx = nSpxPerCol*nSpxPerRow;

if (centroidIdx<nSpx){
int wSpx = width / nSpxPerRow;
int hSpx = height / nSpxPerCol;

int i = centroidIdx / nSpxPerRow;
int j = centroidIdx%nSpxPerRow;

int x = j*wSpx + wSpx / 2;
int y = i*hSpx + hSpx / 2;

float4 color;
surf2Dread(&color, surfFrameLab, x * 16, y);
clusters[centroidIdx] = color.x;
clusters[centroidIdx + nSpx] = color.y;
clusters[centroidIdx + 2 * nSpx] = color.z;
clusters[centroidIdx + 3 * nSpx] = x;
clusters[centroidIdx + 4 * nSpx] = y;
}
}