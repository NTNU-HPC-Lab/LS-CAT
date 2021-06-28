#include "includes.h"
__global__ void cuInsertionSort(float *dist, int dist_pitch, int *ind, int ind_pitch, int width, int height, int k){

// Variables
int l, i, j;
float *p_dist;
int   *p_ind;
float curr_dist, max_dist;
int   curr_row,  max_row;
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

if (xIndex<width){

// Pointer shift, initialization, and max value
p_dist   = dist + xIndex;
p_ind    = ind  + xIndex;
max_dist = p_dist[0];
p_ind[0] = 1;

// Part 1 : sort kth firt elementZ
for (l=1; l<k; l++){
curr_row  = l * dist_pitch;
curr_dist = p_dist[curr_row];
if (curr_dist<max_dist){
i=l-1;
for (int a=0; a<l-1; a++){
if (p_dist[a*dist_pitch]>curr_dist){
i=a;
break;
}
}
for (j=l; j>i; j--){
p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
}
p_dist[i*dist_pitch] = curr_dist;
p_ind[i*ind_pitch]   = l+1;
}
else
p_ind[l*ind_pitch] = l+1;
max_dist = p_dist[curr_row];
}

// Part 2 : insert element in the k-th first lines
max_row = (k-1)*dist_pitch;
for (l=k; l<height; l++){
curr_dist = p_dist[l*dist_pitch];
if (curr_dist<max_dist){
i=k-1;
for (int a=0; a<k-1; a++){
if (p_dist[a*dist_pitch]>curr_dist){
i=a;
break;
}
}
for (j=k-1; j>i; j--){
p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
}
p_dist[i*dist_pitch] = curr_dist;
p_ind[i*ind_pitch]   = l+1;
max_dist             = p_dist[max_row];
}
}
}
}