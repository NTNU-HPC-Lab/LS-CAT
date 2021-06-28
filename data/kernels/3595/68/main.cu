#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "computePointHessian0.cu"
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
float *x = NULL;
cudaMalloc(&x, XSIZE*YSIZE);
float *y = NULL;
cudaMalloc(&y, XSIZE*YSIZE);
float *z = NULL;
cudaMalloc(&z, XSIZE*YSIZE);
int points_num = 1;
int *valid_points = NULL;
cudaMalloc(&valid_points, XSIZE*YSIZE);
int valid_points_num = 1;
double *dh_ang = NULL;
cudaMalloc(&dh_ang, XSIZE*YSIZE);
double *ph93 = NULL;
cudaMalloc(&ph93, XSIZE*YSIZE);
double *ph103 = NULL;
cudaMalloc(&ph103, XSIZE*YSIZE);
double *ph113 = NULL;
cudaMalloc(&ph113, XSIZE*YSIZE);
double *ph123 = NULL;
cudaMalloc(&ph123, XSIZE*YSIZE);
double *ph94 = NULL;
cudaMalloc(&ph94, XSIZE*YSIZE);
double *ph133 = NULL;
cudaMalloc(&ph133, XSIZE*YSIZE);
double *ph104 = NULL;
cudaMalloc(&ph104, XSIZE*YSIZE);
double *ph143 = NULL;
cudaMalloc(&ph143, XSIZE*YSIZE);
double *ph114 = NULL;
cudaMalloc(&ph114, XSIZE*YSIZE);
double *ph153 = NULL;
cudaMalloc(&ph153, XSIZE*YSIZE);
double *ph95 = NULL;
cudaMalloc(&ph95, XSIZE*YSIZE);
double *ph163 = NULL;
cudaMalloc(&ph163, XSIZE*YSIZE);
double *ph105 = NULL;
cudaMalloc(&ph105, XSIZE*YSIZE);
double *ph173 = NULL;
cudaMalloc(&ph173, XSIZE*YSIZE);
double *ph115 = NULL;
cudaMalloc(&ph115, XSIZE*YSIZE);
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
computePointHessian0<<<gridBlock,threadBlock>>>(x,y,z,points_num,valid_points,valid_points_num,dh_ang,ph93,ph103,ph113,ph123,ph94,ph133,ph104,ph143,ph114,ph153,ph95,ph163,ph105,ph173,ph115);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
computePointHessian0<<<gridBlock,threadBlock>>>(x,y,z,points_num,valid_points,valid_points_num,dh_ang,ph93,ph103,ph113,ph123,ph94,ph133,ph104,ph143,ph114,ph153,ph95,ph163,ph105,ph173,ph115);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
computePointHessian0<<<gridBlock,threadBlock>>>(x,y,z,points_num,valid_points,valid_points_num,dh_ang,ph93,ph103,ph113,ph123,ph94,ph133,ph104,ph143,ph114,ph153,ph95,ph163,ph105,ph173,ph115);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}