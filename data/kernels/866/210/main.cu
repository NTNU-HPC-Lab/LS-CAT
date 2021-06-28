#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "mul_sub_grad.cu"
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
float *in1_x = NULL;
cudaMalloc(&in1_x, XSIZE*YSIZE);
float *in1_d = NULL;
cudaMalloc(&in1_d, XSIZE*YSIZE);
float *in2_x = NULL;
cudaMalloc(&in2_x, XSIZE*YSIZE);
float *in2_d = NULL;
cudaMalloc(&in2_d, XSIZE*YSIZE);
float *out = NULL;
cudaMalloc(&out, XSIZE*YSIZE);
int in1ScalarCount = 1;
int in2ScalarCount = 1;
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
mul_sub_grad<<<gridBlock,threadBlock>>>(in1_x,in1_d,in2_x,in2_d,out,in1ScalarCount,in2ScalarCount);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
mul_sub_grad<<<gridBlock,threadBlock>>>(in1_x,in1_d,in2_x,in2_d,out,in1ScalarCount,in2ScalarCount);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
mul_sub_grad<<<gridBlock,threadBlock>>>(in1_x,in1_d,in2_x,in2_d,out,in1ScalarCount,in2ScalarCount);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}