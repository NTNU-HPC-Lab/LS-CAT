#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "bcnn_cuda_axpy_strided_kernel.cu"
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
int n = XSIZE*YSIZE;
int num_batches = 2;
float a = 2;
float *x = NULL;
cudaMalloc(&x, XSIZE*YSIZE);
float *y = NULL;
cudaMalloc(&y, XSIZE*YSIZE);
int dst_stride = 2;
int src_stride = 2;
int x_c = 1;
int x_h = 1;
int x_w = 1;
int y_c = 1;
int y_h = 1;
int y_w = 1;
int min_c = 1;
int min_h = 1;
int min_w = 1;
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
bcnn_cuda_axpy_strided_kernel<<<gridBlock,threadBlock>>>(n,num_batches,a,x,y,dst_stride,src_stride,x_c,x_h,x_w,y_c,y_h,y_w,min_c,min_h,min_w);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
bcnn_cuda_axpy_strided_kernel<<<gridBlock,threadBlock>>>(n,num_batches,a,x,y,dst_stride,src_stride,x_c,x_h,x_w,y_c,y_h,y_w,min_c,min_h,min_w);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
bcnn_cuda_axpy_strided_kernel<<<gridBlock,threadBlock>>>(n,num_batches,a,x,y,dst_stride,src_stride,x_c,x_h,x_w,y_c,y_h,y_w,min_c,min_h,min_w);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}