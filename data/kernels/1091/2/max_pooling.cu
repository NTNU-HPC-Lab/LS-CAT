#include "includes.h"
__global__ void max_pooling(unsigned char* original_img, unsigned char* new_img, unsigned int width, unsigned int num_thread, unsigned int size) {
unsigned int position;
unsigned char max;
for (int i = threadIdx.x; i < size/4; i = i + num_thread) {
position = i + (4 * (i / 4)) + (width * 4 * (i / (width * 2)));
max = original_img[position];
if (original_img[position + 4] > max)
max = original_img[position + 4];
if (original_img[position + width] > max)
max = original_img[position + width];
if (original_img[position + width + 4] > max)
max = original_img[position + width + 1];

new_img[i] = max;
}
}