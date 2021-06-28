#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "kernel.cu"
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
double *Dens = NULL;
cudaMalloc(&Dens, XSIZE*YSIZE);
double *VradInt = NULL;
cudaMalloc(&VradInt, XSIZE*YSIZE);
double *VthetaInt = NULL;
cudaMalloc(&VthetaInt, XSIZE*YSIZE);
double *TemperInt = NULL;
cudaMalloc(&TemperInt, XSIZE*YSIZE);
int nrad = 1;
int nsec = 1;
double *invdiffRmed = NULL;
cudaMalloc(&invdiffRmed, XSIZE*YSIZE);
double *invdiffRsup = NULL;
cudaMalloc(&invdiffRsup, XSIZE*YSIZE);
double *DensInt = NULL;
cudaMalloc(&DensInt, XSIZE*YSIZE);
int Adiabatic = 1;
double *Rmed = NULL;
cudaMalloc(&Rmed, XSIZE*YSIZE);
double dt = 1;
double *VradNew = NULL;
cudaMalloc(&VradNew, XSIZE*YSIZE);
double *VthetaNew = NULL;
cudaMalloc(&VthetaNew, XSIZE*YSIZE);
double *Energy = NULL;
cudaMalloc(&Energy, XSIZE*YSIZE);
double *EnergyInt = NULL;
cudaMalloc(&EnergyInt, XSIZE*YSIZE);
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
kernel<<<gridBlock,threadBlock>>>(Dens,VradInt,VthetaInt,TemperInt,nrad,nsec,invdiffRmed,invdiffRsup,DensInt,Adiabatic,Rmed,dt,VradNew,VthetaNew,Energy,EnergyInt);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
kernel<<<gridBlock,threadBlock>>>(Dens,VradInt,VthetaInt,TemperInt,nrad,nsec,invdiffRmed,invdiffRsup,DensInt,Adiabatic,Rmed,dt,VradNew,VthetaNew,Energy,EnergyInt);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
kernel<<<gridBlock,threadBlock>>>(Dens,VradInt,VthetaInt,TemperInt,nrad,nsec,invdiffRmed,invdiffRsup,DensInt,Adiabatic,Rmed,dt,VradNew,VthetaNew,Energy,EnergyInt);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}