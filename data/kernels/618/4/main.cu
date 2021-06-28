#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "kernel_computeDesctriptorCUDARot.cu"
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
bool *_d_isdescriptor = NULL;
cudaMalloc(&_d_isdescriptor, XSIZE*YSIZE);
char *_d_vdescriptor = NULL;
cudaMalloc(&_d_vdescriptor, XSIZE*YSIZE);
int *_d_keypointsIndexX = NULL;
cudaMalloc(&_d_keypointsIndexX, XSIZE*YSIZE);
int *_d_keypointsIndexY = NULL;
cudaMalloc(&_d_keypointsIndexY, XSIZE*YSIZE);
int *_d_keypointsRotation = NULL;
cudaMalloc(&_d_keypointsRotation, XSIZE*YSIZE);
int _amountofkeypoints = 1;
unsigned int *_d_integralImage = NULL;
cudaMalloc(&_d_integralImage, XSIZE*YSIZE);
int _d_width = XSIZE;
int _d_height = YSIZE;
float _scale = 2;
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
kernel_computeDesctriptorCUDARot<<<gridBlock,threadBlock>>>(_d_isdescriptor,_d_vdescriptor,_d_keypointsIndexX,_d_keypointsIndexY,_d_keypointsRotation,_amountofkeypoints,_d_integralImage,_d_width,_d_height,_scale);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
kernel_computeDesctriptorCUDARot<<<gridBlock,threadBlock>>>(_d_isdescriptor,_d_vdescriptor,_d_keypointsIndexX,_d_keypointsIndexY,_d_keypointsRotation,_amountofkeypoints,_d_integralImage,_d_width,_d_height,_scale);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
kernel_computeDesctriptorCUDARot<<<gridBlock,threadBlock>>>(_d_isdescriptor,_d_vdescriptor,_d_keypointsIndexX,_d_keypointsIndexY,_d_keypointsRotation,_amountofkeypoints,_d_integralImage,_d_width,_d_height,_scale);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}