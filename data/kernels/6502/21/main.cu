#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "sd_t_s1_2_kernel.cu"
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
size_t h1d = 1;
size_t h2d = 1;
size_t h3d = 1;
size_t p4d = 1;
size_t p6d = 1;
size_t p4ld_t2 = 1;
size_t h1ld_t2 = 1;
size_t h3ld_v2 = 1;
size_t h2ld_v2 = 1;
size_t p6ld_v2 = 1;
size_t h3ld_t3 = 1;
size_t h2ld_t3 = 1;
size_t h1ld_t3 = 1;
size_t p6ld_t3 = 1;
size_t p4ld_t3 = 1;
double *t2_d = NULL;
cudaMalloc(&t2_d, XSIZE*YSIZE);
double *v2_d = NULL;
cudaMalloc(&v2_d, XSIZE*YSIZE);
size_t p4 = 1;
size_t total_x = 1;
double *t3d = NULL;
cudaMalloc(&t3d, XSIZE*YSIZE);
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
sd_t_s1_2_kernel<<<gridBlock,threadBlock>>>(h1d,h2d,h3d,p4d,p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,t2_d,v2_d,p4,total_x,t3d);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
sd_t_s1_2_kernel<<<gridBlock,threadBlock>>>(h1d,h2d,h3d,p4d,p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,t2_d,v2_d,p4,total_x,t3d);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
sd_t_s1_2_kernel<<<gridBlock,threadBlock>>>(h1d,h2d,h3d,p4d,p6d,p4ld_t2,h1ld_t2,h3ld_v2,h2ld_v2,p6ld_v2,h3ld_t3,h2ld_t3,h1ld_t3,p6ld_t3,p4ld_t3,t2_d,v2_d,p4,total_x,t3d);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}