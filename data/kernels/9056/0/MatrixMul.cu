#include "includes.h"
#define THREADS_PER_BLOCK 256






__global__ void MatrixMul( float *Md , float *Nd , float *Pd , const int WIDTH )
{



int COL = threadIdx.x + blockIdx.x * blockDim.x;
int ROW = threadIdx.y + blockIdx.y * blockDim.y;



if (ROW < WIDTH && COL < WIDTH) {
for (int i = 0; i < WIDTH; i++) {
Pd[ROW * WIDTH + COL] += Md[ROW * WIDTH + i] * Nd [i * WIDTH + COL];
}
}

}