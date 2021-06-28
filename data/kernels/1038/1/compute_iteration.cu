#include "includes.h"
__global__ void compute_iteration(char* buffer, char* out_buffer, size_t pitch, size_t pitch_out, int width, int height)
{
const int x = blockDim.x * blockIdx.x + threadIdx.x;
const int y = blockDim.y * blockIdx.y + threadIdx.y;
if (x >= width || y >= height)
return;

int left_x = (x - 1 + width) % width;
int right_x = (x + 1) % width;
int up_y = (y - 1 + height) % height;
int down_y = (y + 1) % height;
char n_alive = buffer[up_y * pitch + left_x] + buffer[up_y * pitch + x]
+ buffer[up_y * pitch + right_x] + buffer[y * pitch + left_x]
+ buffer[y * pitch + right_x] + buffer[down_y * pitch + left_x]
+ buffer[down_y * pitch + x] + buffer[down_y * pitch + right_x];

out_buffer[y * pitch + x] =
n_alive == 3 || (buffer[y * pitch + x] && n_alive == 2);
}