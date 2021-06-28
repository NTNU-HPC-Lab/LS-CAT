#include "includes.h"
__global__ void _softback(int nrows, int ncols, float *y, float *dy) {
/* y is layer output, i.e. unnormalized log probabilities.
On output y will contain normalized probabilities.
Conceptually this is a forward calculation but we do it here for efficiency.
dy is the label matrix: each column is a one-hot vector indicating the correct label.
On output dy will be the gradient of softmax loss wrt probabilities.
*/
float y0, sum;
int i0, i1;
int col = threadIdx.x + blockIdx.x * blockDim.x;
while (col < ncols) {
i0 = col * nrows;
i1 = i0  + nrows;
y0 = -INFINITY;
//y0 = y[i0];
for (int i=i0; i<i1; i++) {
if (y[i] > y0) {
y0 = y[i];
}
}
sum = 0;
for (int i=i0; i<i1; i++) {
y[i] = exp(y[i]-y0);
sum += y[i];
}
for (int i=i0; i<i1; i++) {
y[i] /= sum;
dy[i] = (y[i] - dy[i]) / ncols;
}
col += blockDim.x * gridDim.x;
}
}