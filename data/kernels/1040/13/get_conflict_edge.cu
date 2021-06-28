#include "includes.h"
__device__ void get_conflict_edge(bool *dl_matrix, short *deleted_rows, int *row_group, const int conflict_node_id, const int search_depth, int *conflict_edge, const int vertex_num, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num) {
//*conflict_col_id = 0;
// int idxa = 0;
// int idxb = 0;

for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
// find the conflict edge that connects current node and the most closest
// node.
if (deleted_rows[i] == -conflict_node_id) {
atomicMax(conflict_edge, i);
}
if (row_group[i] == search_depth + 1 &&
deleted_rows[i] == conflict_node_id) {
atomicMax(conflict_edge + 1, i);
}
}
}
__global__ void get_conflict_edge(int *dl_matrix, int *deleted_rows, int *deleted_cols, int *row_group, const int conflict_node_id, const int search_depth, int *conflict_edge, const int vertex_num, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num) {
//*conflict_col_id = 0;
//int idxa = 0;
//int idxb = 0;

for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
// find the conflict edge that connects current node and the most closest
// node.
if (deleted_rows[i] == -conflict_node_id) {
atomicMax(conflict_edge, i);
}
if (row_group[i] == search_depth + 1 &&
deleted_rows[i] == conflict_node_id) {
atomicMax(conflict_edge+1, i);
}
}
__syncthreads();
}