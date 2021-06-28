#include "includes.h"
__global__ void labeling(const char *text, int *pos, int text_size){
int index = threadIdx.x*blockDim.y+threadIdx.y + blockDim.x*blockDim.y*(gridDim.y*blockIdx.x + blockIdx.y);
if (index >= text_size) {
return;
}
pos[index] = 0;
if (text[index] <= ' ')
return ;
for (int k = index; k >= 0; k--) {
if (text[k] <= ' ') {
pos[index] = index - k;
return;
}
}
pos[index] = index+1;

}