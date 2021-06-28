#include "includes.h"
__global__ void add_img(float *image_padded, float *ave1, float *ave2, int nx, int ny, int nima) {

// Block index
int bx = blockIdx.x;

// Thread index
int tx = threadIdx.x;

float sum1 = 0.0;
float sum2 = 0.0;
int index = tx+bx*nx;
int index2 = tx+(nx>>1)+(bx+(ny>>1))*(nx*2+2);

for (int i=0; i<nima; i+=2) sum1 += image_padded[index2+i*(nx*2+2)*ny*2];
for (int i=1; i<nima; i+=2) sum2 += image_padded[index2+i*(nx*2+2)*ny*2];
ave1[index] = sum1;
ave2[index] = sum2;

return;
}