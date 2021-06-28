#include "includes.h"

#define tileSize 32

//function for data initialization
void initialization( double *M,  double *N, int arow, int acol, int brow, int bcol);
//(for Debugging) prints out the input data
void printInput( double *M,  double *N, int arow, int acol, int brow,  int bcol);
//(for Debugging) prints out the output data
void printOutput( double *P_C,  double *P_G, int arow, int bcol);

//GPU kernels




__global__
__global__ void matrixMultiplication(double* M, double* N, double* P, int widthAHeightB, int heightA, int widthB) {

int Mstart=widthAHeightB*tileSize*blockIdx.y;
int Mend=Mstart+ widthAHeightB - 1;
int mstep=tileSize;
int Nstart=tileSize*blockIdx.x;
int nstep=tileSize*widthB;
double temp=0;

__shared__ double Ms[tileSize][tileSize];
__shared__ double Ns[tileSize][tileSize];

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