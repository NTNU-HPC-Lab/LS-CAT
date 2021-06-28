#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "histDupeKernel.cu"
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
const float *data1 = NULL;
cudaMalloc(&data1, XSIZE*YSIZE);
const float *data2 = NULL;
cudaMalloc(&data2, XSIZE*YSIZE);
const float *confidence1 = NULL;
cudaMalloc(&confidence1, XSIZE*YSIZE);
const float *confidence2 = NULL;
cudaMalloc(&confidence2, XSIZE*YSIZE);
int *ids1 = NULL;
cudaMalloc(&ids1, XSIZE*YSIZE);
int *ids2 = NULL;
cudaMalloc(&ids2, XSIZE*YSIZE);
int *results_id1 = NULL;
cudaMalloc(&results_id1, XSIZE*YSIZE);
int *results_id2 = NULL;
cudaMalloc(&results_id2, XSIZE*YSIZE);
float *results_similarity = NULL;
cudaMalloc(&results_similarity, XSIZE*YSIZE);
int *result_count = NULL;
cudaMalloc(&result_count, XSIZE*YSIZE);
const int N1 = 1;
const int N2 = 1;
const int max_results = 1;
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
histDupeKernel<<<gridBlock,threadBlock>>>(data1,data2,confidence1,confidence2,ids1,ids2,results_id1,results_id2,results_similarity,result_count,N1,N2,max_results);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
histDupeKernel<<<gridBlock,threadBlock>>>(data1,data2,confidence1,confidence2,ids1,ids2,results_id1,results_id2,results_similarity,result_count,N1,N2,max_results);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
histDupeKernel<<<gridBlock,threadBlock>>>(data1,data2,confidence1,confidence2,ids1,ids2,results_id1,results_id2,results_similarity,result_count,N1,N2,max_results);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}