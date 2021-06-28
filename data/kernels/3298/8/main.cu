#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "read_G_matrix_kernel.cu"
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
int S = 1;
int vertex_index = 1;
int *i_index = NULL;
cudaMalloc(&i_index, XSIZE*YSIZE);
int *j_index = NULL;
cudaMalloc(&j_index, XSIZE*YSIZE);
bool *is_Bennett = NULL;
cudaMalloc(&is_Bennett, XSIZE*YSIZE);
double *exp_Vj = NULL;
cudaMalloc(&exp_Vj, XSIZE*YSIZE);
double *N_ptr = NULL;
cudaMalloc(&N_ptr, XSIZE*YSIZE);
int LD_N = 1;
double *G_ptr = NULL;
cudaMalloc(&G_ptr, XSIZE*YSIZE);
int LD_G = 1;
double *result_ptr = NULL;
cudaMalloc(&result_ptr, XSIZE*YSIZE);
int incr = 1;
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
read_G_matrix_kernel<<<gridBlock,threadBlock>>>(S,vertex_index,i_index,j_index,is_Bennett,exp_Vj,N_ptr,LD_N,G_ptr,LD_G,result_ptr,incr);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
read_G_matrix_kernel<<<gridBlock,threadBlock>>>(S,vertex_index,i_index,j_index,is_Bennett,exp_Vj,N_ptr,LD_N,G_ptr,LD_G,result_ptr,incr);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
read_G_matrix_kernel<<<gridBlock,threadBlock>>>(S,vertex_index,i_index,j_index,is_Bennett,exp_Vj,N_ptr,LD_N,G_ptr,LD_G,result_ptr,incr);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}