#include "includes.h"
__global__ void FillAdjacencyMatrix(float* adj_mat , float* maskBuffer , int size , int cols , int rows ,int Nsegs){
int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
int icol = idx % cols;
int irow = idx / cols;
int seg_id1=-1;
if (idx<size){
if (icol<cols-2 && irow<rows-2 && irow>1 && icol>1){
seg_id1 = maskBuffer[idx];
if (seg_id1!=maskBuffer[idx+1]){
adj_mat[ (int)maskBuffer[idx+1] + seg_id1*Nsegs ]=1;
adj_mat[ seg_id1 + Nsegs*(int)maskBuffer[idx+1] ]=1; /// it can happen that a->b, but b->a wont appear...
}
else if (seg_id1!=maskBuffer[idx-cols]){
adj_mat[ (int)maskBuffer[idx-cols] + seg_id1*Nsegs ]=1;
adj_mat[ seg_id1 + Nsegs*(int)maskBuffer[idx-cols] ]=1; /// it can happen that a->b, but b->a wont appear...
}
}
}
}