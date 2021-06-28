#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 1e-4

__global__ void matting_laplacian_kernel( float *input, float *grad, int h, int w, int *CSR_rowIdx, int *CSR_colIdx, float *CSR_val, int N )
{
int size = h * w;
int _id = blockIdx.x * blockDim.x + threadIdx.x;

if (_id < size) {
int x = _id % w, y = _id / w;
int id = x * h + y;

/// Because matting laplacian L is systematic, sum row is sufficient
// 1.1 Binary search
int start = 0;
int end = N-1;
int mid = (start + end)/2;
int index = -1;
while (start <= end) {
int rowIdx = (CSR_rowIdx[mid]) - 1;

if (rowIdx == id) {
index = mid;    break;
}
if (rowIdx > id) {
end = mid - 1;
mid = (start + end)/2;
} else {
start = mid + 1;
mid = (start + end)/2;
}
}
if (index != -1) {
// 1.2 Complete range
int index_s = index, index_e = index;
while ( index_s >= 0 && ((CSR_rowIdx[index_s] - 1) == id) )
index_s--;
while ( index_e <  N && ((CSR_rowIdx[index_e] - 1) == id) )
index_e++;
// 1.3 Sum this row
for (int i = index_s + 1; i < index_e; i++) {
//int rowIdx = CSR_rowIdx[i] - 1;
int _colIdx = (CSR_colIdx[i]) - 1;
float val  = CSR_val[i];

int _x = _colIdx / h, _y = _colIdx % h;
int colIdx = _y *w + _x;

grad[_id] 			+= 2*val * input[colIdx];
grad[_id + size] 	+= 2*val * input[colIdx + size];
grad[_id + 2*size]  += 2*val * input[colIdx + 2*size];
}

}
}

return ;
}