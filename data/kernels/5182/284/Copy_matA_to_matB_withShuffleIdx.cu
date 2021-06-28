#include "includes.h"
__global__  void Copy_matA_to_matB_withShuffleIdx (float * A , float * B , int size, int cols , float * new_idxs, int max_rows){
int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
int irow = id / cols;
int icol = id % cols;
if (id<size){
int irow_new = max_rows - 1 - irow; /// it was ascending, so I need to revert it...
int irow_old = new_idxs[irow];
B[irow_new*cols + icol] = A[irow_old*cols + icol];
}
}