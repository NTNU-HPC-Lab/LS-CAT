#include "includes.h"
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
//We need to iterate with tiles - starting point and end needed for tiles
int Mstart=Width*BLOCK_SIZE*blockIdx.y;//rows of matrix M
int Mend=Mstart+Width-1;
int mstep=BLOCK_SIZE;
int Nstart=BLOCK_SIZE*blockIdx.x;//cols of matrix N
int nstep=BLOCK_SIZE*Width;
float temp=0;

//loop through tiles


for(int m=Mstart,n=Nstart;m<Mend;m+=mstep,n+=nstep){
__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];
Ms[threadIdx.y][threadIdx.x]=M[m+Width*threadIdx.y+threadIdx.x];
Ns[threadIdx.y][threadIdx.x]=N[n+Width*threadIdx.y+threadIdx.x];
__syncthreads();


for (int i = 0; i < BLOCK_SIZE; ++i) {
temp += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
}

__syncthreads();

}

P[Width * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x + Width * threadIdx.y + threadIdx.x] = temp;

}