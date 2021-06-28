#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "copy_kernel_tobuf.cu"
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
char *dest = NULL;
cudaMalloc(&dest, XSIZE*YSIZE);
char *src = NULL;
cudaMalloc(&src, XSIZE*YSIZE);
int rx_s = 1;
int rx_e = 1;
int ry_s = 1;
int ry_e = 1;
int rz_s = 1;
int rz_e = 1;
int x_step = 1;
int y_step = 1;
int z_step = 1;
int size_x = XSIZE*YSIZE;
int size_y = XSIZE*YSIZE;
int size_z = XSIZE*YSIZE;
int buf_strides_x = 2;
int buf_strides_y = 2;
int buf_strides_z = 2;
int type_size = XSIZE*YSIZE;
int dim = 2;
int OPS_soa = 1;
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
copy_kernel_tobuf<<<gridBlock,threadBlock>>>(dest,src,rx_s,rx_e,ry_s,ry_e,rz_s,rz_e,x_step,y_step,z_step,size_x,size_y,size_z,buf_strides_x,buf_strides_y,buf_strides_z,type_size,dim,OPS_soa);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
copy_kernel_tobuf<<<gridBlock,threadBlock>>>(dest,src,rx_s,rx_e,ry_s,ry_e,rz_s,rz_e,x_step,y_step,z_step,size_x,size_y,size_z,buf_strides_x,buf_strides_y,buf_strides_z,type_size,dim,OPS_soa);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
copy_kernel_tobuf<<<gridBlock,threadBlock>>>(dest,src,rx_s,rx_e,ry_s,ry_e,rz_s,rz_e,x_step,y_step,z_step,size_x,size_y,size_z,buf_strides_x,buf_strides_y,buf_strides_z,type_size,dim,OPS_soa);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}