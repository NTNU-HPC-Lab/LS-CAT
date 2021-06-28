#include "includes.h"
__global__ void matmul_double(double* A, double* B ,  double* C, int M, int N, int K)
{



int bx = blockIdx.x ;
int by = blockIdx.y ;

int tx = threadIdx.x ;
int ty = threadIdx.y ;

int row = by * TILE_WIDTH + ty ;
int col = bx * TILE_WIDTH + tx ;

__shared__ double SA[TILE_WIDTH][TILE_WIDTH+1] ;
__shared__ double SB[TILE_WIDTH][TILE_WIDTH+1] ;

double Csub = 0;



for (int i = 0; i < (K-1)/TILE_WIDTH +1 ; ++i)
{
/* code */


if ( (row < M) && (i * TILE_WIDTH + tx < K ) ){
SA[ty][tx] = A[row*K + i * TILE_WIDTH + tx] ;
}
else{
SA[ty][tx] = 0;
}




if ( (col < N ) &&  ( i * TILE_WIDTH + ty < K)){
SB[tx][ty] = B[ col * K + i*TILE_WIDTH + ty] ;
}
else{
SB[tx][ty] = 0;
}


__syncthreads() ;

for (int k = 0; k < TILE_WIDTH; ++k){
Csub += SA[ty][k]*SB[tx][k] ;
}

__syncthreads() ;


}

//C[row*n + col] = Csub ;

if ( (row < M ) && ( col < N )){
C[ row * N + col] = Csub ;
}



}