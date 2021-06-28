#include "includes.h"
__device__ void remove_cols(short *deleted_cols, int *col_group, const int conflict_col_id, const int total_dl_matrix_col_num) {
for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
if (col_group[i] == col_group[conflict_col_id]) {
deleted_cols[i] = -1;
}
}
}
__global__ void remove_cols(int *deleted_cols, int *col_group, const int conflict_col_id, const int total_dl_matrix_col_num) {
for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
if (col_group[i] == col_group[conflict_col_id]) {
deleted_cols[i] = -1;
}
}
}