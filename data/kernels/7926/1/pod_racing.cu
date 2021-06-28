#include "includes.h"
__global__ void pod_racing(unsigned int *d_rand, unsigned int *win, unsigned int *loss, unsigned int size, int *iter) {
int index = threadIdx.x + blockDim.x*blockIdx.x;
const unsigned int flips[] = { 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1 };
if (index < size) {
//printf("%d ", iter[0]);
if ((d_rand[index] % 2) != flips[iter[0]]) {
iter[0] = 0;
loss[index] = 1;
//printf("loss ");
}
else {
iter[0] = iter[0] + 1;
if (iter[0] == 15) {
win[index] = 1;
iter[0] = 0;
//printf("win ");
}
}
}
}