#include "includes.h"
__device__ void compute_conv(int row, int col, double2 *d_c, double *d_a, double2 *d_b, int *o_row_vect, int *o_col_vect, int ma, int na, int mb, int nb, int mc, int nc) {

int count_row = o_row_vect[row];
int count_col = o_col_vect[col];
int row_idx;
int col_idx;
int k_row_idx;
int k_col_idx;
int k_col_start_idx;

int i_row_idx;
int i_col_idx;
int i_col_start_idx;

k_row_idx = row - (ma - 1);
k_row_idx = k_row_idx < 0 ? 0:k_row_idx;

k_col_start_idx = col - (na - 1);
k_col_start_idx = k_col_start_idx < 1? 0: k_col_start_idx;
k_col_idx = k_col_start_idx;

i_row_idx = row > (ma - 1) ? (ma - 1) : row;
i_col_idx = col > (na - 1) ? (na - 1) : col;
i_col_start_idx = i_col_idx;

for ( row_idx = 0; row_idx < count_row; row_idx++) {
for (col_idx = 0; col_idx < count_col; col_idx++) {

d_c[col + nc * row].x += d_a[i_col_idx + na * i_row_idx] * d_b[k_col_idx + nb * k_row_idx].x;
d_c[col + nc * row].y += d_a[i_col_idx + na * i_row_idx] * d_b[k_col_idx + nb * k_row_idx].y;

k_col_idx++;
i_col_idx--;
}
k_row_idx++;
i_row_idx--;
k_col_idx = k_col_start_idx;
i_col_idx = i_col_start_idx;

}
}
__global__ void kernel_conv(double2 *d_c, double *d_a, double2 *d_b, int *d_row_vect, int *d_col_vect, int ma, int na, int mb, int nb, int mc, int nc) {

int i, idx;
int rownum, colnum, num_threads;

idx = threadIdx.x + blockIdx.x * blockDim.x;
num_threads = gridDim.x * blockDim.x;

for(i=idx; i< (mc *nc); i=i+num_threads){

rownum = i / nc;
colnum = i % nc;

// Device Function call to multiply the Image pixel with the Kernel Image pixel and perform addition
compute_conv(rownum, colnum, d_c, d_a, d_b, d_row_vect, d_col_vect, ma, na, mb, nb, mc, nc);
}
}