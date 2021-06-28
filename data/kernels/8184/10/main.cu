#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "d_sct_axinterp.cu"
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
float *sct3d = NULL;
cudaMalloc(&sct3d, XSIZE*YSIZE);
const float *scts1 = NULL;
cudaMalloc(&scts1, XSIZE*YSIZE);
const int4 *sctaxR = NULL;
cudaMalloc(&sctaxR, XSIZE*YSIZE);
const float4 *sctaxW = NULL;
cudaMalloc(&sctaxW, XSIZE*YSIZE);
const short *sn1_sn11 = NULL;
cudaMalloc(&sn1_sn11, XSIZE*YSIZE);
int NBIN = 1;
int NSN1 = 1;
int SPN = 1;
int offtof = 1;
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
d_sct_axinterp<<<gridBlock,threadBlock>>>(sct3d,scts1,sctaxR,sctaxW,sn1_sn11,NBIN,NSN1,SPN,offtof);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
d_sct_axinterp<<<gridBlock,threadBlock>>>(sct3d,scts1,sctaxR,sctaxW,sn1_sn11,NBIN,NSN1,SPN,offtof);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
d_sct_axinterp<<<gridBlock,threadBlock>>>(sct3d,scts1,sctaxR,sctaxW,sn1_sn11,NBIN,NSN1,SPN,offtof);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}