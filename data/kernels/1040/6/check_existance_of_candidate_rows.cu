#include "includes.h"
__device__ void check_existance_of_candidate_rows( short *deleted_rows, int *row_group, const int search_depth, int *token, int *selected_row_id, const int total_dl_matrix_row_num) {
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
// std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
if (deleted_rows[i] == 0 && row_group[i] == search_depth) {
// std::cout<<"Candidate Row Found...."<<std::endl;
// atomicExch(token, 1);
*token = 1;
atomicMin(selected_row_id, i);
// If find a number can break;
// break;
}
}
}
__global__ void check_existance_of_candidate_rows( int *deleted_rows, int *row_group, const int search_depth, int *token, int *selected_row_id, const int total_dl_matrix_row_num) {
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i = i + blockDim.x) {
// std::cout<<deleted_rows[i]<<' '<<row_group[i]<<std::endl;
if (deleted_rows[i] == 0 && row_group[i] == search_depth) {
// std::cout<<"Candidate Row Found...."<<std::endl;
atomicExch(token, 1);
atomicMin(selected_row_id, i);
}
}
__syncthreads();
}