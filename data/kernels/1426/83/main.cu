#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "computeScoreGradientList.cu"
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
float *trans_x = NULL;
cudaMalloc(&trans_x, XSIZE*YSIZE);
float *trans_y = NULL;
cudaMalloc(&trans_y, XSIZE*YSIZE);
float *trans_z = NULL;
cudaMalloc(&trans_z, XSIZE*YSIZE);
int *valid_points = NULL;
cudaMalloc(&valid_points, XSIZE*YSIZE);
int *starting_voxel_id = NULL;
cudaMalloc(&starting_voxel_id, XSIZE*YSIZE);
int *voxel_id = NULL;
cudaMalloc(&voxel_id, XSIZE*YSIZE);
int valid_points_num = 1;
double *centroid_x = NULL;
cudaMalloc(&centroid_x, XSIZE*YSIZE);
double *centroid_y = NULL;
cudaMalloc(&centroid_y, XSIZE*YSIZE);
double *centroid_z = NULL;
cudaMalloc(&centroid_z, XSIZE*YSIZE);
int voxel_num = 1;
double *e_x_cov_x = NULL;
cudaMalloc(&e_x_cov_x, XSIZE*YSIZE);
double *cov_dxd_pi = NULL;
cudaMalloc(&cov_dxd_pi, XSIZE*YSIZE);
double gauss_d1 = 1;
int valid_voxel_num = 1;
double *score_gradients = NULL;
cudaMalloc(&score_gradients, XSIZE*YSIZE);
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
computeScoreGradientList<<<gridBlock,threadBlock>>>(trans_x,trans_y,trans_z,valid_points,starting_voxel_id,voxel_id,valid_points_num,centroid_x,centroid_y,centroid_z,voxel_num,e_x_cov_x,cov_dxd_pi,gauss_d1,valid_voxel_num,score_gradients);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
computeScoreGradientList<<<gridBlock,threadBlock>>>(trans_x,trans_y,trans_z,valid_points,starting_voxel_id,voxel_id,valid_points_num,centroid_x,centroid_y,centroid_z,voxel_num,e_x_cov_x,cov_dxd_pi,gauss_d1,valid_voxel_num,score_gradients);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
computeScoreGradientList<<<gridBlock,threadBlock>>>(trans_x,trans_y,trans_z,valid_points,starting_voxel_id,voxel_id,valid_points_num,centroid_x,centroid_y,centroid_z,voxel_num,e_x_cov_x,cov_dxd_pi,gauss_d1,valid_voxel_num,score_gradients);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}