#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "kContract.cu"
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
float *expanded_data = NULL;
cudaMalloc(&expanded_data, XSIZE*YSIZE);
float *targets = NULL;
cudaMalloc(&targets, XSIZE*YSIZE);
int num_images = 1;
int num_input_channels = 1;
int image_size_y = XSIZE*YSIZE;
int image_size_x = XSIZE*YSIZE;
int num_modules_y = 1;
int num_modules_x = 1;
int kernel_size_y = XSIZE*YSIZE;
int kernel_size_x = XSIZE*YSIZE;
int padding_y = 1;
int padding_x = 1;
int stride_y = 2;
int stride_x = 2;
int num_modules_batch = 2;
int module_id_offset = 1;
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
kContract<<<gridBlock,threadBlock>>>(expanded_data,targets,num_images,num_input_channels,image_size_y,image_size_x,num_modules_y,num_modules_x,kernel_size_y,kernel_size_x,padding_y,padding_x,stride_y,stride_x,num_modules_batch,module_id_offset);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
kContract<<<gridBlock,threadBlock>>>(expanded_data,targets,num_images,num_input_channels,image_size_y,image_size_x,num_modules_y,num_modules_x,kernel_size_y,kernel_size_x,padding_y,padding_x,stride_y,stride_x,num_modules_batch,module_id_offset);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
kContract<<<gridBlock,threadBlock>>>(expanded_data,targets,num_images,num_input_channels,image_size_y,image_size_x,num_modules_y,num_modules_x,kernel_size_y,kernel_size_x,padding_y,padding_x,stride_y,stride_x,num_modules_batch,module_id_offset);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}