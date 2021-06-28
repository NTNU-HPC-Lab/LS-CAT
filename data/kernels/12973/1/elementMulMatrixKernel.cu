#include "includes.h"
__global__ void elementMulMatrixKernel(double *dev_w, const double *dev_U, const double *dev_V, unsigned int index_row_i, unsigned int index_column_j, unsigned int dim1_U, unsigned int dim1_V)
{
//---------------------------------------------------------------------------------------------------------
// determine indices - row first
//---------------------------------------------------------------------------------------------------------

// 1D
//int idx_k = threadIdx.x;
unsigned int idx_k = blockIdx.x * gridDim.x + threadIdx.x;

// check index range to abort
if (idx_k > dim1_U-1)
return;

unsigned int idx_u_i0 = index_row_i * dim1_U;
unsigned int idx_v_0j = index_column_j;

unsigned int idx_u_ik = idx_u_i0 + idx_k;
unsigned int idx_v_kj = idx_v_0j + idx_k*dim1_V;

//---------------------------------------------------------------------------------------------------------

do
{
//---------------------------------------------------------------------------------------------------------
// access the arrays - row major
//---------------------------------------------------------------------------------------------------------

dev_w[idx_k] = dev_U[idx_u_ik] * dev_V[idx_v_kj];

//---------------------------------------------------------------------------------------------------------
// determine new indices - row first
//---------------------------------------------------------------------------------------------------------

// 1D
//idx_k += blockIdx.x;
idx_k += blockIdx.x * gridDim.x;

idx_u_ik = idx_u_i0 + idx_k;
idx_v_kj = idx_v_0j + idx_k*dim1_V;

} while (idx_k < dim1_U);
//---------------------------------------------------------------------------------------------------------

}