#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "simKernel.cu"
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
int N_stgy = 1;
int N_batch = 2;
double *alpha = NULL;
cudaMalloc(&alpha, XSIZE*YSIZE);
double *mid = NULL;
cudaMalloc(&mid, XSIZE*YSIZE);
double *gap = NULL;
cudaMalloc(&gap, XSIZE*YSIZE);
int *late = NULL;
cudaMalloc(&late, XSIZE*YSIZE);
int *pos = NULL;
cudaMalloc(&pos, XSIZE*YSIZE);
int *rest_lag = NULL;
cudaMalloc(&rest_lag, XSIZE*YSIZE);
double *prof = NULL;
cudaMalloc(&prof, XSIZE*YSIZE);
double *last_prc = NULL;
cudaMalloc(&last_prc, XSIZE*YSIZE);
int *cnt = NULL;
cudaMalloc(&cnt, XSIZE*YSIZE);
double fee = 1;
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
simKernel<<<gridBlock,threadBlock>>>(N_stgy,N_batch,alpha,mid,gap,late,pos,rest_lag,prof,last_prc,cnt,fee);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
simKernel<<<gridBlock,threadBlock>>>(N_stgy,N_batch,alpha,mid,gap,late,pos,rest_lag,prof,last_prc,cnt,fee);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
simKernel<<<gridBlock,threadBlock>>>(N_stgy,N_batch,alpha,mid,gap,late,pos,rest_lag,prof,last_prc,cnt,fee);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}