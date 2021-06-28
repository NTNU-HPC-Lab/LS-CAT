#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "convert2DVectorToAngleMagnitude_kernel.cu"
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
uchar4 *d_angle_image = NULL;
cudaMalloc(&d_angle_image, XSIZE*YSIZE);
uchar4 *d_magnitude_image = NULL;
cudaMalloc(&d_magnitude_image, XSIZE*YSIZE);
float *d_vector_X = NULL;
cudaMalloc(&d_vector_X, XSIZE*YSIZE);
float *d_vector_Y = NULL;
cudaMalloc(&d_vector_Y, XSIZE*YSIZE);
int width = XSIZE;
int height = YSIZE;
float lower_ang = 1;
float upper_ang = 1;
float lower_mag = 1;
float upper_mag = 1;
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
convert2DVectorToAngleMagnitude_kernel<<<gridBlock,threadBlock>>>(d_angle_image,d_magnitude_image,d_vector_X,d_vector_Y,width,height,lower_ang,upper_ang,lower_mag,upper_mag);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
convert2DVectorToAngleMagnitude_kernel<<<gridBlock,threadBlock>>>(d_angle_image,d_magnitude_image,d_vector_X,d_vector_Y,width,height,lower_ang,upper_ang,lower_mag,upper_mag);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
convert2DVectorToAngleMagnitude_kernel<<<gridBlock,threadBlock>>>(d_angle_image,d_magnitude_image,d_vector_X,d_vector_Y,width,height,lower_ang,upper_ang,lower_mag,upper_mag);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}