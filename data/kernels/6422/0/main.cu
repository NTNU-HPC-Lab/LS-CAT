#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "check_collisions.cu"
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
float x1_robot = 1;
float y1_robot = 1;
float x2_robot = 1;
float y2_robot = 1;
float *x1_obs = NULL;
cudaMalloc(&x1_obs, XSIZE*YSIZE);
float *y1_obs = NULL;
cudaMalloc(&y1_obs, XSIZE*YSIZE);
float *x2_obs = NULL;
cudaMalloc(&x2_obs, XSIZE*YSIZE);
float *y2_obs = NULL;
cudaMalloc(&y2_obs, XSIZE*YSIZE);
bool *collisions = NULL;
cudaMalloc(&collisions, XSIZE*YSIZE);
int *indexes = NULL;
cudaMalloc(&indexes, XSIZE*YSIZE);
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
check_collisions<<<gridBlock,threadBlock>>>(x1_robot,y1_robot,x2_robot,y2_robot,x1_obs,y1_obs,x2_obs,y2_obs,collisions,indexes);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
check_collisions<<<gridBlock,threadBlock>>>(x1_robot,y1_robot,x2_robot,y2_robot,x1_obs,y1_obs,x2_obs,y2_obs,collisions,indexes);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
check_collisions<<<gridBlock,threadBlock>>>(x1_robot,y1_robot,x2_robot,y2_robot,x1_obs,y1_obs,x2_obs,y2_obs,collisions,indexes);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}