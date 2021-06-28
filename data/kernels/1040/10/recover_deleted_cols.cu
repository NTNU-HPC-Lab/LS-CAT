#include "includes.h"
__device__ void recover_deleted_cols(short *deleted_cols, const int search_depth, const int total_dl_matrix_col_num) {
for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
if (deleted_cols[i] >= search_depth) {
deleted_cols[i] = 0;
}
}
}
__global__ void recover_deleted_cols(int *deleted_cols, const int search_depth, const int total_dl_matrix_col_num) {
for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
if (deleted_cols[i] >= search_depth) {
deleted_cols[i] = 0;
}
}
}