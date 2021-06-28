#include "includes.h"
//#define DEBUG
//#define HANDLE_ERROR(x) if((x) != 0) cout << "Error!" << endl;

using namespace std;

struct SubBlock{

int * nnz_global_i_idx;
int * nnz_global_o_idx;

int nnz;
int * nnz_local_r_idx;
int * nnz_local_c_idx;
float * nnz_values;
};
//void printSubBlocksInfo(SubBlock * sbs, int nsbs, int mem_b_size);




__global__ void CudaCompute(SubBlock * d_sbs, float * d_x, float * d_y, int nblocks, int mem_b_size, int nrows, int ncols , float * sub_y_arr){
/*
sub_y_arr stores float number, with nblocks rows, mem_b_size columns
*/
//#ifdef DEBUG
//printf("This is Cuda Block # %d: \n", blockIdx.x);
//#endif

//if(blockIdx.x >= nblocks)
//    return;


//SubBlock * work_sb = &d_sbs[blockIdx.x];


//printSubBlocksInfo(work_sb, 1, mem_b_size);

/*
float * x_sub = (float *) malloc(mem_b_size * sizeof(float));
float * y_sub = (float *) malloc(mem_b_size * sizeof(float));
//float * x;


for(int i = 0; i < mem_b_size; i++){
if(work_sb->nnz_global_i_idx[i] > 0 && work_sb->nnz_global_i_idx[i] <= ncols){
// d_x   indexing starts from '1'
// x_sub indexing starts from '0'
x_sub[i] = d_x[work_sb->nnz_global_i_idx[i] - 1];
}
else{
x_sub[i] = 0.0;
}
}

for(int i = 0; i < work_sb->nnz; i++){
int x_sub_idx = work_sb->nnz_local_c_idx[i] - 1;
int y_sub_idx = work_sb->nnz_local_r_idx[i] - 1;
y_sub[y_sub_idx] += work_sb->nnz_values[i] * x_sub[x_sub_idx];
//#ifdef DEBUG
//    printf("This is Cuda Block # %d:  Computing (%d, %d) product as (%f)\n", blockIdx.x, x_sub_idx, y_sub_idx, work_sb->nnz_values[i] * x_sub[x_sub_idx]);
//#endif
}

for(int i = 0; i < mem_b_size; i++){
sub_y_arr[blockIdx.x * mem_b_size + i] = y_sub[i];
}
*/

}