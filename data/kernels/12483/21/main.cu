#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "set_chunk_data.cu"
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
int x = 1;
int y = 1;
double dx = 1;
double dy = 1;
double *cell_x = NULL;
cudaMalloc(&cell_x, XSIZE*YSIZE);
double *cell_y = NULL;
cudaMalloc(&cell_y, XSIZE*YSIZE);
double *cell_dx = NULL;
cudaMalloc(&cell_dx, XSIZE*YSIZE);
double *cell_dy = NULL;
cudaMalloc(&cell_dy, XSIZE*YSIZE);
double *vertex_x = NULL;
cudaMalloc(&vertex_x, XSIZE*YSIZE);
double *vertex_y = NULL;
cudaMalloc(&vertex_y, XSIZE*YSIZE);
double *volume = NULL;
cudaMalloc(&volume, XSIZE*YSIZE);
double *x_area = NULL;
cudaMalloc(&x_area, XSIZE*YSIZE);
double *y_area = NULL;
cudaMalloc(&y_area, XSIZE*YSIZE);
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
set_chunk_data<<<gridBlock,threadBlock>>>(x,y,dx,dy,cell_x,cell_y,cell_dx,cell_dy,vertex_x,vertex_y,volume,x_area,y_area);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
set_chunk_data<<<gridBlock,threadBlock>>>(x,y,dx,dy,cell_x,cell_y,cell_dx,cell_dy,vertex_x,vertex_y,volume,x_area,y_area);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
set_chunk_data<<<gridBlock,threadBlock>>>(x,y,dx,dy,cell_x,cell_y,cell_dx,cell_dy,vertex_x,vertex_y,volume,x_area,y_area);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}