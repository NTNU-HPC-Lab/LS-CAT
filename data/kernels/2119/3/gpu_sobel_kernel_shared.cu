#include "includes.h"
__global__ void gpu_sobel_kernel_shared(u_char *Source, u_char *Resultat, unsigned width, unsigned height) {
__shared__ u_char tuile[BLOCKDIM_X][BLOCKDIM_Y];

int x = threadIdx.x;
int y = threadIdx.y;
int i = blockIdx.y*(BLOCKDIM_Y-2) + y;
int j = blockIdx.x*(BLOCKDIM_X-2) + x;

int globalIndex = i*width+j;

if ((i==0)||(i>=height-1)||(j==0)||(j>=width-1)) {}
else {
//mainstream
tuile[x][y] = Source[globalIndex];
__syncthreads();

u_char val;
if ((x>0)&&(y>0)&&(x<BLOCKDIM_X-1)&&(y<BLOCKDIM_Y-1)) {
val = std::abs(tuile[x-1][y-1] + tuile[x-1][y] + tuile[x-1][y+1] -\
(tuile[x+1][y-1] + tuile[x+1][y] + tuile[x+1][y+1]));
Resultat[globalIndex]  = val + std::abs(tuile[x-1][y-1] + tuile[x][y-1] + tuile[x+1][y-1] -\
(tuile[x-1][y+1] + tuile[x][y+1] + tuile[x+1][y+1]));
}
}
}