#include "includes.h"
__global__ void rectify(unsigned char * original_img, unsigned char* new_img, unsigned int num_thread, unsigned int size) {
for (int i = threadIdx.x; i < size; i = i + num_thread) {
if (original_img[i] < 127)
new_img[i] = 127;
else
new_img[i] = original_img[i];
}
}