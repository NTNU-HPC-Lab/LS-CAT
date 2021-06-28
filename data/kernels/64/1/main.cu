#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "copyPixelsInSlicesRGB.cu"
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
float *ptrinput0 = NULL;
cudaMalloc(&ptrinput0, XSIZE*YSIZE);
float *ptrkslices0 = NULL;
cudaMalloc(&ptrkslices0, XSIZE*YSIZE);
int dH = 1;
int dW = 1;
int kH = 1;
int kW = 1;
int size1 = XSIZE*YSIZE;
int size2 = XSIZE*YSIZE;
int isize1 = XSIZE*YSIZE;
int isize2 = XSIZE*YSIZE;
int nInputPlane = 1;
int padleft = 1;
int padright = 1;
int padup = 1;
int paddown = 1;
int inputstr0 = 1;
int kslicesstr0 = 1;
int batchsize = XSIZE*YSIZE;
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
copyPixelsInSlicesRGB<<<gridBlock,threadBlock>>>(ptrinput0,ptrkslices0,dH,dW,kH,kW,size1,size2,isize1,isize2,nInputPlane,padleft,padright,padup,paddown,inputstr0,kslicesstr0,batchsize);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
copyPixelsInSlicesRGB<<<gridBlock,threadBlock>>>(ptrinput0,ptrkslices0,dH,dW,kH,kW,size1,size2,isize1,isize2,nInputPlane,padleft,padright,padup,paddown,inputstr0,kslicesstr0,batchsize);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
copyPixelsInSlicesRGB<<<gridBlock,threadBlock>>>(ptrinput0,ptrkslices0,dH,dW,kH,kW,size1,size2,isize1,isize2,nInputPlane,padleft,padright,padup,paddown,inputstr0,kslicesstr0,batchsize);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}