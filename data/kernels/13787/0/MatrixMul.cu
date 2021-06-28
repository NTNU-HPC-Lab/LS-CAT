#include "includes.h"
/*CUDA 2-D Matrix Multiplication*/


#define TILE_WIDTH 2
#define WIDTH  100


// main routine
__global__ void MatrixMul( float *A_d , float *B_d , float *C_d)
{
// calculate thread id
unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

C_d[row*WIDTH+col] = 0;
for (int k = 0 ; k<WIDTH ; k++ )
{
C_d[row*WIDTH + col]+= A_d[row * WIDTH + k ] * B_d[ k * WIDTH + col] ;
}
}