#include "includes.h"
__global__ void merge(unsigned char * img_all, unsigned char * img, float * selection, int n, int stride)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;
int idx = 0;
float sum = 0;
float weight = 0;
for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
sum = 0;
for (idx = 0; idx < n; idx ++) sum += selection[idx * stride + ((y+j)*width + x)];

for (idx = 0; idx < n; idx ++) selection[idx * stride + ((y+j)*width + x)] /= (sum + 1e-5);

img[3*((y+j)*width + x)] = 0;
img[3*((y+j)*width + x)+1] = 0;
img[3*((y+j)*width + x)+2] = 0;

for (idx = 0; idx < n; idx ++) {

//weight = selection[idx * stride + ((y+j)*width + x)];
weight = 0.25;
//weight = 0.5;


img[3*((y+j)*width + x)] += (unsigned char) (img_all[idx * stride * 3 + 3*((y+j)*width + x)] * weight);
img[3*((y+j)*width + x)+1] += (unsigned char) (img_all[idx * stride * 3 + 3*((y+j)*width + x) + 1] * weight);
img[3*((y+j)*width + x)+2] += (unsigned char)(img_all[idx * stride * 3 + 3*((y+j)*width + x) + 2] * weight);

}


}
}