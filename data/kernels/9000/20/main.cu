#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "Corrector_gpu.cu"
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
double GTIME = 1;
double *local_time = NULL;
cudaMalloc(&local_time, XSIZE*YSIZE);
double *step = NULL;
cudaMalloc(&step, XSIZE*YSIZE);
int *next = NULL;
cudaMalloc(&next, XSIZE*YSIZE);
unsigned long nextsize = 1;
double4 *pos_CH = NULL;
cudaMalloc(&pos_CH, XSIZE*YSIZE);
double4 *vel_CH = NULL;
cudaMalloc(&vel_CH, XSIZE*YSIZE);
double4 *a_tot_D = NULL;
cudaMalloc(&a_tot_D, XSIZE*YSIZE);
double4 *a1_tot_D = NULL;
cudaMalloc(&a1_tot_D, XSIZE*YSIZE);
double4 *a2_tot_D = NULL;
cudaMalloc(&a2_tot_D, XSIZE*YSIZE);
double4 *a_H0 = NULL;
cudaMalloc(&a_H0, XSIZE*YSIZE);
double4 *a3_H = NULL;
cudaMalloc(&a3_H, XSIZE*YSIZE);
double ETA6 = 1;
double ETA4 = 1;
double DTMAX = 1;
double DTMIN = 1;
unsigned int N = 1;
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
Corrector_gpu<<<gridBlock,threadBlock>>>(GTIME,local_time,step,next,nextsize,pos_CH,vel_CH,a_tot_D,a1_tot_D,a2_tot_D,a_H0,a3_H,ETA6,ETA4,DTMAX,DTMIN,N);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
Corrector_gpu<<<gridBlock,threadBlock>>>(GTIME,local_time,step,next,nextsize,pos_CH,vel_CH,a_tot_D,a1_tot_D,a2_tot_D,a_H0,a3_H,ETA6,ETA4,DTMAX,DTMIN,N);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
Corrector_gpu<<<gridBlock,threadBlock>>>(GTIME,local_time,step,next,nextsize,pos_CH,vel_CH,a_tot_D,a1_tot_D,a2_tot_D,a_H0,a3_H,ETA6,ETA4,DTMAX,DTMIN,N);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}