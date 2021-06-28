#include "includes.h"
__global__ void MatrixMulVarSharedMemoryKernel(float* M, float* N, float* P, int widthAHeightB, int heightA, int widthB) {

int Mstart=widthAHeightB*tileSize*blockIdx.y;
int Mend=Mstart+ widthAHeightB - 1;
int mstep=tileSize;
int Nstart=tileSize*blockIdx.x;
int nstep=tileSize*widthB;
float temp=0;

__shared__ float Ms[tileSize][tileSize];
__shared__ float Ns[tileSize][tileSize];

//area where the tiles fits without "cutting"
if(Mstart < (heightA/tileSize)*tileSize*widthAHeightB && Nstart%widthB < (widthB/tileSize)*tileSize ){
for(int m=Mstart,n=Nstart;m<Mend;m+=mstep,n+=nstep){
Ms[threadIdx.y][threadIdx.x]=M[m+widthAHeightB*threadIdx.y+threadIdx.x];
Ns[threadIdx.y][threadIdx.x]=N[n+widthB*threadIdx.y+threadIdx.x];
__syncthreads();


for (int i = 0; i < tileSize; ++i) {
temp += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
}
__syncthreads();

}
} else {//the rest of the matrix
for(int m=Mstart,n=Nstart;m<=Mend;m+=mstep,n+=nstep){

if(m%widthAHeightB + threadIdx.x < widthAHeightB && blockIdx.y*tileSize + threadIdx.y < heightA){
Ms[threadIdx.y][threadIdx.x]=M[m+widthAHeightB*threadIdx.y+threadIdx.x];
}
else{
Ms[threadIdx.y][threadIdx.x]=0.0;
}

if((n/widthB) + threadIdx.y < widthAHeightB && blockIdx.x*tileSize + threadIdx.x < widthB){
Ns[threadIdx.y][threadIdx.x]=N[n+widthB*threadIdx.y+threadIdx.x];
}
else{
Ns[threadIdx.y][threadIdx.x]=0.0;
}
__syncthreads();


for (int i = 0; i < tileSize; ++i) {
temp += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
}
__syncthreads();

}
}



if(blockIdx.y*tileSize + threadIdx.y < heightA && blockIdx.x*tileSize + threadIdx.x < widthB){
P[widthB * tileSize * blockIdx.y + tileSize * blockIdx.x + widthB * threadIdx.y + threadIdx.x] = temp;
}
}