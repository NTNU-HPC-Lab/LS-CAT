#include "includes.h"
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// **** 	A = M x N		****			AxB=C
//****		B = N x K		****
//**** 	C = M x K		****
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


static const int M = 3;
static const int N = 5;
static const int K = 4;
static const int TILE_WIDTH = 2;

using namespace std;
//---------------------------------------------------------------------------------
/**
* This macro checks return value of the CUDA runtime call and exits
* the application if the call failed.
*/
__global__ void MatrixMulKernel(int ARows,int ACols, int BRows, int BCols, int CRows, int CCols,unsigned int* A_d, unsigned int *B_d, unsigned int *C_d) {

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// **** Populate matrixMultiplication kernel function ****
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


int CValue = 0;

int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

__shared__ int As[TILE_WIDTH][TILE_WIDTH];
__shared__ int Bs[TILE_WIDTH][TILE_WIDTH];

for (int k = 0; k < (TILE_WIDTH + ACols - 1)/TILE_WIDTH; k++) {

if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows)
As[threadIdx.y][threadIdx.x] = A_d[Row*ACols + k*TILE_WIDTH + threadIdx.x];
else
As[threadIdx.y][threadIdx.x] = 0;

if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols)
Bs[threadIdx.y][threadIdx.x] = B_d[(k*TILE_WIDTH + threadIdx.y)*BCols + Col];
else
Bs[threadIdx.y][threadIdx.x] = 0;

__syncthreads();

for (int n = 0; n < TILE_WIDTH; ++n)
CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

__syncthreads();
}

if (Row < CRows && Col < CCols)
C_d[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
(blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;



}