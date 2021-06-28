#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "MHDUpdatePrim_CUDA3_kernel.cu"
#include<chrono>
#include<iostream>
using namespace std;
using namespace std::chrono;
int blocks_[20][2] = {{8,8},{16,16},{24,24},{32,32},{1,64},{1,128},{1,192},{1,256},{1,320},{1,384},{1,448},{1,512},{1,576},{1,640},{1,704},{1,768},{1,832},{1,896},{1,960},{1,1024}};
int matrices_[7][2] = {{240,240},{496,496},{784,784},{1016,1016},{1232,1232},{1680,1680},{2024,2024}};
int main(int argc, char **argv) {
cudaSetDevice(0);
char* p;int matrix_len=strtol(argv[1], &p, 10);
for(int matrix_looper=0;matrix_looper<matrix_len;matrix_looper++){
for(int block_looper=0;block_looper<20;block_looper++){
int XSIZE=matrices_[matrix_looper][0],YSIZE=matrices_[matrix_looper][1],BLOCKX=blocks_[block_looper][0],BLOCKY=blocks_[block_looper][1];
float *Rho = NULL;
cudaMalloc(&Rho, XSIZE*YSIZE);
float *Vx = NULL;
cudaMalloc(&Vx, XSIZE*YSIZE);
float *Vy = NULL;
cudaMalloc(&Vy, XSIZE*YSIZE);
float *Vz = NULL;
cudaMalloc(&Vz, XSIZE*YSIZE);
float *Etot = NULL;
cudaMalloc(&Etot, XSIZE*YSIZE);
float *Bx = NULL;
cudaMalloc(&Bx, XSIZE*YSIZE);
float *By = NULL;
cudaMalloc(&By, XSIZE*YSIZE);
float *Bz = NULL;
cudaMalloc(&Bz, XSIZE*YSIZE);
float *Phi = NULL;
cudaMalloc(&Phi, XSIZE*YSIZE);
float *dUD = NULL;
cudaMalloc(&dUD, XSIZE*YSIZE);
float *dUS1 = NULL;
cudaMalloc(&dUS1, XSIZE*YSIZE);
float *dUS2 = NULL;
cudaMalloc(&dUS2, XSIZE*YSIZE);
float *dUS3 = NULL;
cudaMalloc(&dUS3, XSIZE*YSIZE);
float *dUTau = NULL;
cudaMalloc(&dUTau, XSIZE*YSIZE);
float *dUBx = NULL;
cudaMalloc(&dUBx, XSIZE*YSIZE);
float *dUBy = NULL;
cudaMalloc(&dUBy, XSIZE*YSIZE);
float *dUBz = NULL;
cudaMalloc(&dUBz, XSIZE*YSIZE);
float *dUPhi = NULL;
cudaMalloc(&dUPhi, XSIZE*YSIZE);
float dt = 1;
float C_h = 1;
float C_p = 1;
int size = XSIZE*YSIZE;
int iXSIZE= XSIZE;
int iYSIZE= YSIZE;
while(iXSIZE%BLOCKX!=0)
{
iXSIZE++;
}
while(iYSIZE%BLOCKY!=0)
{
iYSIZE++;
}
dim3 gridBlock(iXSIZE/BLOCKX, iYSIZE/BLOCKY);
dim3 threadBlock(BLOCKX, BLOCKY);
cudaFree(0);
MHDUpdatePrim_CUDA3_kernel<<<gridBlock,threadBlock>>>(Rho,Vx,Vy,Vz,Etot,Bx,By,Bz,Phi,dUD,dUS1,dUS2,dUS3,dUTau,dUBx,dUBy,dUBz,dUPhi,dt,C_h,C_p,size);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
MHDUpdatePrim_CUDA3_kernel<<<gridBlock,threadBlock>>>(Rho,Vx,Vy,Vz,Etot,Bx,By,Bz,Phi,dUD,dUS1,dUS2,dUS3,dUTau,dUBx,dUBy,dUBz,dUPhi,dt,C_h,C_p,size);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
MHDUpdatePrim_CUDA3_kernel<<<gridBlock,threadBlock>>>(Rho,Vx,Vy,Vz,Etot,Bx,By,Bz,Phi,dUD,dUS1,dUS2,dUS3,dUTau,dUBx,dUBy,dUBz,dUPhi,dt,C_h,C_p,size);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}