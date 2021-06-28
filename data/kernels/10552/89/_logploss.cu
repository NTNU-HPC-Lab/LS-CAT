#include "includes.h"
__global__ void _logploss(int nrows, int ncols, float *y, float *dy) {
/* Similar to softmaxloss, except y is assumed normalized logp and is not overwritten.
y is layer output, i.e. normalized log probabilities.
dy is the label matrix: each column is a one-hot vector indicating the correct label.
On output dy will be the gradient of softmax loss wrt log probabilities.
*/
int col = threadIdx.x + blockIdx.x * blockDim.x;
int i0, i1;
while (col < ncols) {
i0 = col * nrows;
i1 = i0  + nrows;
for (int i=i0; i<i1; i++) {
dy[i] = (exp(y[i]) - dy[i]) / ncols;
}
col += blockDim.x * gridDim.x;
}
}