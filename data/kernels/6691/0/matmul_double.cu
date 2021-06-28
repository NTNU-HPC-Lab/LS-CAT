#include "includes.h"
/* objective
* 	C = A*B  // A[m][k], B[k][n], C[m][n]
* compile: nvcc --gpu-architecture=compute_60 --gpu-code=sm_60 -O3 matmul_double.cu -o matmul_double
Using nvprof for this lab

nvprof -- query-metrics
nvprof dram_read_transactions ./test 1024 1024  128
nvprof ./test 1024 1024 128

second line of result shows time for GPU kernel

GFlop   ( 2MNK * 10^-9 ) / time (second)

*/



#define TILE_WIDTH 16

__global__ void matmul_double(double* A, double* B , double* C, int M, int N, int K)
{
/// complete code

int bx = blockIdx.x ;
int by = blockIdx.y ;

int tx = threadIdx.x ;
int ty = threadIdx.y ;

int row = by * TILE_WIDTH + ty ;
int col = bx * TILE_WIDTH + tx ;

__shared__ double SA[TILE_WIDTH][TILE_WIDTH] ;
__shared__ double SB[TILE_WIDTH][TILE_WIDTH] ;

double Csub = 0;

for (int i = 0; i < (K-1)/TILE_WIDTH +1 ; ++i)
{
/* code */
//SA[ty][tx] = A[row*n + i * TILE_WIDTH + tx] ;
//SB[ty][tx] = B[(i * TILE_WIDTH + ty )*n + col   ] ;

if ( (row < M) && (i * TILE_WIDTH + tx < K ) ){
SA[ty][tx] = A[row*K + i * TILE_WIDTH + tx] ;
}
else{
SA[ty][tx] = 0;
}

if ( (col < N ) && ( i * TILE_WIDTH + ty < K) ){
SB[ty][tx] = B[(i*TILE_WIDTH + ty)*N + col] ;
}
else{
SB[ty][tx] = 0;
}



__syncthreads() ;

for (int k = 0; k < TILE_WIDTH; ++k){
Csub += SA[ty][k] * SB[k][tx] ;
}

__syncthreads() ;


}

//C[row*n + col] = Csub ;

if ( (row < M ) && ( col < N )){
C[ row * N + col] = Csub ;
}



}