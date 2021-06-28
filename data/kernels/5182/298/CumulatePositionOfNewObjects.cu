#include "includes.h"
__global__ void CumulatePositionOfNewObjects(float* mask , float* maskNewIds , float* maskOut, int mask_size, int mask_cols, float* centers, int centers_size, int centers_columns){
int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
int icol = idx % mask_cols;
int irow = idx / mask_cols;

int i_mask, i_obj;

if (idx<mask_size){
i_mask = mask[idx];
i_obj  = maskNewIds[i_mask];
maskOut[idx] = i_obj;
if (i_obj*centers_columns+2<centers_size){
atomicAdd(centers + 0 + i_obj*centers_columns , (float)icol);
atomicAdd(centers + 1 + i_obj*centers_columns , (float)irow);
atomicAdd(centers + 2 + i_obj*centers_columns , 1.0f);
}
}
}