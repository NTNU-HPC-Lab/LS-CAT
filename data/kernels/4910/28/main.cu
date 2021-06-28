#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "ComputePositions.cu"
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
float *g_Data1 = NULL;
cudaMalloc(&g_Data1, XSIZE*YSIZE);
float *g_Data2 = NULL;
cudaMalloc(&g_Data2, XSIZE*YSIZE);
float *g_Data3 = NULL;
cudaMalloc(&g_Data3, XSIZE*YSIZE);
int *d_Ptrs = NULL;
cudaMalloc(&d_Ptrs, XSIZE*YSIZE);
float *d_Sift = NULL;
cudaMalloc(&d_Sift, XSIZE*YSIZE);
int numPts = 1;
int maxPts = 1;
int w = XSIZE;
int h = YSIZE;
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
ComputePositions<<<gridBlock,threadBlock>>>(g_Data1,g_Data2,g_Data3,d_Ptrs,d_Sift,numPts,maxPts,w,h);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
ComputePositions<<<gridBlock,threadBlock>>>(g_Data1,g_Data2,g_Data3,d_Ptrs,d_Sift,numPts,maxPts,w,h);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
ComputePositions<<<gridBlock,threadBlock>>>(g_Data1,g_Data2,g_Data3,d_Ptrs,d_Sift,numPts,maxPts,w,h);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}