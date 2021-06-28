#include "includes.h"
__device__ void get_conflict_col_id(bool *dl_matrix, short *deleted_cols, int *conflict_col_id, int *conflict_edge, int total_dl_matrix_col_num, int vertex_num) {
// if(threadIdx.x==0){
//  printf("conflict edge a %d edge b
//  %d\n",conflict_edge[0],conflict_edge[1]);
// }
bool *edge_a_dlmatrix =
dl_matrix + conflict_edge[0] * total_dl_matrix_col_num;
bool *edge_b_dlmatrix =
dl_matrix + conflict_edge[1] * total_dl_matrix_col_num;
for (int j = threadIdx.x; j < total_dl_matrix_col_num; j = j + blockDim.x) {
if (edge_a_dlmatrix[j] == edge_b_dlmatrix[j] && deleted_cols[j] > 0 &&
edge_b_dlmatrix[j] == 1) {
atomicMax(conflict_col_id, j);
}
}
}
__global__ void get_conflict_col_id(int *dl_matrix, int *deleted_cols, int *conflict_col_id, int *conflict_edge, int total_dl_matrix_col_num, int vertex_num){
//if(threadIdx.x==0){
//  printf("conflict edge a %d edge b %d\n",conflict_edge[0],conflict_edge[1]);
// }
for (int j = threadIdx.x; j < total_dl_matrix_col_num;
j = j + blockDim.x) {
if (dl_matrix[conflict_edge[0] * total_dl_matrix_col_num + j]
== dl_matrix[conflict_edge[1] * total_dl_matrix_col_num + j] &&
deleted_cols[j] > 0 && dl_matrix[conflict_edge[1] * total_dl_matrix_col_num + j]==1) {
atomicMax(conflict_col_id, j);
}
}
__syncthreads();
}