#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "exclscnmb2e.cu"
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
int *d_data0 = NULL;
cudaMalloc(&d_data0, XSIZE*YSIZE);
int *d_output0 = NULL;
cudaMalloc(&d_output0, XSIZE*YSIZE);
int *d_data1 = NULL;
cudaMalloc(&d_data1, XSIZE*YSIZE);
int *d_output1 = NULL;
cudaMalloc(&d_output1, XSIZE*YSIZE);
int *d_data2 = NULL;
cudaMalloc(&d_data2, XSIZE*YSIZE);
int *d_output2 = NULL;
cudaMalloc(&d_output2, XSIZE*YSIZE);
int *d_data3 = NULL;
cudaMalloc(&d_data3, XSIZE*YSIZE);
int *d_output3 = NULL;
cudaMalloc(&d_output3, XSIZE*YSIZE);
int *d_data4 = NULL;
cudaMalloc(&d_data4, XSIZE*YSIZE);
int *d_output4 = NULL;
cudaMalloc(&d_output4, XSIZE*YSIZE);
int *d_data5 = NULL;
cudaMalloc(&d_data5, XSIZE*YSIZE);
int *d_output5 = NULL;
cudaMalloc(&d_output5, XSIZE*YSIZE);
int *d_data6 = NULL;
cudaMalloc(&d_data6, XSIZE*YSIZE);
int *d_output6 = NULL;
cudaMalloc(&d_output6, XSIZE*YSIZE);
int *d_data7 = NULL;
cudaMalloc(&d_data7, XSIZE*YSIZE);
int *d_output7 = NULL;
cudaMalloc(&d_output7, XSIZE*YSIZE);
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
exclscnmb2e<<<gridBlock,threadBlock>>>(d_data0,d_output0,d_data1,d_output1,d_data2,d_output2,d_data3,d_output3,d_data4,d_output4,d_data5,d_output5,d_data6,d_output6,d_data7,d_output7);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
exclscnmb2e<<<gridBlock,threadBlock>>>(d_data0,d_output0,d_data1,d_output1,d_data2,d_output2,d_data3,d_output3,d_data4,d_output4,d_data5,d_output5,d_data6,d_output6,d_data7,d_output7);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
exclscnmb2e<<<gridBlock,threadBlock>>>(d_data0,d_output0,d_data1,d_output1,d_data2,d_output2,d_data3,d_output3,d_data4,d_output4,d_data5,d_output5,d_data6,d_output6,d_data7,d_output7);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}