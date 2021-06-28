#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "aux_fields.cu"
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
double *V = NULL;
cudaMalloc(&V, XSIZE*YSIZE);
double *K = NULL;
cudaMalloc(&K, XSIZE*YSIZE);
double gdt = 1;
double dt = 1;
double *Ax = NULL;
cudaMalloc(&Ax, XSIZE*YSIZE);
double *Ay = NULL;
cudaMalloc(&Ay, XSIZE*YSIZE);
double *Az = NULL;
cudaMalloc(&Az, XSIZE*YSIZE);
double *px = NULL;
cudaMalloc(&px, XSIZE*YSIZE);
double *py = NULL;
cudaMalloc(&py, XSIZE*YSIZE);
double *pz = NULL;
cudaMalloc(&pz, XSIZE*YSIZE);
double *pAx = NULL;
cudaMalloc(&pAx, XSIZE*YSIZE);
double *pAy = NULL;
cudaMalloc(&pAy, XSIZE*YSIZE);
double *pAz = NULL;
cudaMalloc(&pAz, XSIZE*YSIZE);
double2 *GV = NULL;
cudaMalloc(&GV, XSIZE*YSIZE);
double2 *EV = NULL;
cudaMalloc(&EV, XSIZE*YSIZE);
double2 *GK = NULL;
cudaMalloc(&GK, XSIZE*YSIZE);
double2 *EK = NULL;
cudaMalloc(&EK, XSIZE*YSIZE);
double2 *GpAx = NULL;
cudaMalloc(&GpAx, XSIZE*YSIZE);
double2 *GpAy = NULL;
cudaMalloc(&GpAy, XSIZE*YSIZE);
double2 *GpAz = NULL;
cudaMalloc(&GpAz, XSIZE*YSIZE);
double2 *EpAx = NULL;
cudaMalloc(&EpAx, XSIZE*YSIZE);
double2 *EpAy = NULL;
cudaMalloc(&EpAy, XSIZE*YSIZE);
double2 *EpAz = NULL;
cudaMalloc(&EpAz, XSIZE*YSIZE);
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
aux_fields<<<gridBlock,threadBlock>>>(V,K,gdt,dt,Ax,Ay,Az,px,py,pz,pAx,pAy,pAz,GV,EV,GK,EK,GpAx,GpAy,GpAz,EpAx,EpAy,EpAz);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
aux_fields<<<gridBlock,threadBlock>>>(V,K,gdt,dt,Ax,Ay,Az,px,py,pz,pAx,pAy,pAz,GV,EV,GK,EK,GpAx,GpAy,GpAz,EpAx,EpAy,EpAz);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
aux_fields<<<gridBlock,threadBlock>>>(V,K,gdt,dt,Ax,Ay,Az,px,py,pz,pAx,pAy,pAz,GV,EV,GK,EK,GpAx,GpAy,GpAz,EpAx,EpAy,EpAz);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}