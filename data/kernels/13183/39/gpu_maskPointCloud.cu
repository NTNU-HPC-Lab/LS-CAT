#include "includes.h"
__global__ void gpu_maskPointCloud(float4* verts, const int width, const int height, const int* mask) {

const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x >= width || y >= height)
return;

const int index = x + y*width;

int m = mask[index];
if (m == 0) {
verts[index].w = -1;
}

}