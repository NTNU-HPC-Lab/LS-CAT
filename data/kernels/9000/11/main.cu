#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "dev_get_gravity_at_point.cu"
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
float eps2 = 1;
float *eps = NULL;
cudaMalloc(&eps, XSIZE*YSIZE);
float *xh = NULL;
cudaMalloc(&xh, XSIZE*YSIZE);
float *yh = NULL;
cudaMalloc(&yh, XSIZE*YSIZE);
float *zh = NULL;
cudaMalloc(&zh, XSIZE*YSIZE);
float *xt = NULL;
cudaMalloc(&xt, XSIZE*YSIZE);
float *yt = NULL;
cudaMalloc(&yt, XSIZE*YSIZE);
float *zt = NULL;
cudaMalloc(&zt, XSIZE*YSIZE);
float *ax = NULL;
cudaMalloc(&ax, XSIZE*YSIZE);
float *ay = NULL;
cudaMalloc(&ay, XSIZE*YSIZE);
float *az = NULL;
cudaMalloc(&az, XSIZE*YSIZE);
int n = XSIZE*YSIZE;
float *field_m = NULL;
cudaMalloc(&field_m, XSIZE*YSIZE);
float *fxh = NULL;
cudaMalloc(&fxh, XSIZE*YSIZE);
float *fyh = NULL;
cudaMalloc(&fyh, XSIZE*YSIZE);
float *fzh = NULL;
cudaMalloc(&fzh, XSIZE*YSIZE);
float *fxt = NULL;
cudaMalloc(&fxt, XSIZE*YSIZE);
float *fyt = NULL;
cudaMalloc(&fyt, XSIZE*YSIZE);
float *fzt = NULL;
cudaMalloc(&fzt, XSIZE*YSIZE);
int n_field = 1;
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
dev_get_gravity_at_point<<<gridBlock,threadBlock>>>(eps2,eps,xh,yh,zh,xt,yt,zt,ax,ay,az,n,field_m,fxh,fyh,fzh,fxt,fyt,fzt,n_field);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
dev_get_gravity_at_point<<<gridBlock,threadBlock>>>(eps2,eps,xh,yh,zh,xt,yt,zt,ax,ay,az,n,field_m,fxh,fyh,fzh,fxt,fyt,fzt,n_field);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
dev_get_gravity_at_point<<<gridBlock,threadBlock>>>(eps2,eps,xh,yh,zh,xt,yt,zt,ax,ay,az,n,field_m,fxh,fyh,fzh,fxt,fyt,fzt,n_field);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}