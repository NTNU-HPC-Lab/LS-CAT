#include "includes.h"
__device__ void get_vertex_row_group(int *row_group, bool *dl_matrix, const int vertex_num, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num) {
// printf("%d %d\n", vertex_num, total_dl_matrix_row_num);
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
for (int j = 0, cur_index = i * total_dl_matrix_col_num; j < vertex_num;
j++, cur_index++) {
row_group[i] += (int)(dl_matrix[cur_index]) * (j + 1);
}
}
}
__global__ void get_vertex_row_group(int *row_group, int *dl_matrix, const int vertex_num, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num) {
// printf("%d %d\n", vertex_num, total_dl_matrix_row_num);
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
for (int j = 0; j < vertex_num; j++) {
row_group[i] += dl_matrix[i * total_dl_matrix_col_num + j] * (j + 1);
}
}
}