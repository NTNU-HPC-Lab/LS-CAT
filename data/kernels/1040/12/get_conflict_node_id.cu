#include "includes.h"
__device__ void get_conflict_node_id(short *deleted_rows, int *row_group, const int search_depth, int *conflict_node_id, const int total_dl_matrix_row_num) {
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
if (row_group[i] == search_depth + 1 &&
deleted_rows[i] < search_depth + 1) {
atomicMax(conflict_node_id, deleted_rows[i]);
}
}
}
__global__ void get_conflict_node_id(int *deleted_rows, int *row_group, const int search_depth, int *conflict_node_id, const int total_dl_matrix_row_num) {
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
if (row_group[i] == search_depth + 1 &&  deleted_rows[i] < search_depth+1) {
atomicMax(conflict_node_id, deleted_rows[i]);
}
}
__syncthreads();
}