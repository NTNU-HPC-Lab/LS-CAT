#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "init_vertex_group.cu"
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
int *row_group = NULL;
cudaMalloc(&row_group, XSIZE*YSIZE);
bool *dl_matrix = NULL;
cudaMalloc(&dl_matrix, XSIZE*YSIZE);
int *vertex_num = NULL;
cudaMalloc(&vertex_num, XSIZE*YSIZE);
int *t_cn = NULL;
cudaMalloc(&t_cn, XSIZE*YSIZE);
int *t_rn = NULL;
cudaMalloc(&t_rn, XSIZE*YSIZE);
int *offset_row = NULL;
cudaMalloc(&offset_row, XSIZE*YSIZE);
int *offset_matrix = NULL;
cudaMalloc(&offset_matrix, XSIZE*YSIZE);
int graph_count = 1;
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
init_vertex_group<<<gridBlock,threadBlock>>>(row_group,dl_matrix,vertex_num,t_cn,t_rn,offset_row,offset_matrix,graph_count);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
init_vertex_group<<<gridBlock,threadBlock>>>(row_group,dl_matrix,vertex_num,t_cn,t_rn,offset_row,offset_matrix,graph_count);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
init_vertex_group<<<gridBlock,threadBlock>>>(row_group,dl_matrix,vertex_num,t_cn,t_rn,offset_row,offset_matrix,graph_count);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}