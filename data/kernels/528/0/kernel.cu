#include "includes.h"

using namespace std;

//Check for edges valid to be part of augmented path

//Update frontier

__global__ void kernel(bool* adj_mat, const int N, bool* visited, int* frontier, bool* new_frontier, bool* par_mat, int* cap_mat, int* cap_max_mat) {
int row_idx = frontier[blockIdx.x+1];
long offset = N * row_idx;

int col_idx = threadIdx.x;
long offset2 = N * col_idx;
if(adj_mat[offset + col_idx] && (cap_mat[offset + col_idx] < cap_max_mat[offset + col_idx]) && !visited[col_idx]) {
new_frontier[col_idx] = true;
par_mat[offset2 + row_idx] = true;
}

if(adj_mat[offset2 + row_idx] && (cap_mat[offset2 + row_idx] > 0) && !visited[col_idx]) {
new_frontier[col_idx] = true;
par_mat[offset2 + row_idx] = true;
}
}