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
__global__ void init_vertex_group(int *row_group, bool *dl_matrix, int *vertex_num, int *t_cn, int *t_rn, int *offset_row, int *offset_matrix, int graph_count) {
int k = blockIdx.x;
if (k < graph_count) {
get_vertex_row_group(row_group + offset_row[k],
dl_matrix + offset_matrix[k], vertex_num[k], t_rn[k],
t_cn[k]);
}
}