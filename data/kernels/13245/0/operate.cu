#include "includes.h"

#define SEED
#define BLOCK_SIZE 32

typedef struct _data {
char * values;
char * next_values;
int width;
int height;
} data;

__global__ void operate(char * source, char * goal, int sizex, int sizey) {
__shared__ char local[BLOCK_SIZE + MASK_WIDTH - 1][BLOCK_SIZE + MASK_WIDTH - 1];
int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

int index = i * sizex + j;

int prim_x = j - MASK_RADIUS;
int first_x = prim_x;
for(; first_x - prim_x + threadIdx.x < MASK_WIDTH + BLOCK_SIZE - 1; first_x += BLOCK_SIZE) {
int prim_y = i - MASK_RADIUS;
int first_y = prim_y;
for(; first_y - prim_y + threadIdx.y < MASK_WIDTH + BLOCK_SIZE - 1; first_y += BLOCK_SIZE) {
if(first_y >= 0 && first_y < sizey && first_x >= 0 && first_x < sizex) {
local[first_y - prim_y + threadIdx.y][first_x - prim_x + threadIdx.x] =
source[first_y * sizex + first_x];
}
else {
local[first_y - prim_y + threadIdx.y][first_x - prim_x + threadIdx.x] = '0';
}
}
}
__syncthreads();

if(i < sizey && j < sizex) {
int l_j, l_i;
int amount = 0;

for(l_i = 0; l_i < MASK_WIDTH; l_i++) {
if( ( (int) threadIdx.y + l_i >= 0 ) && ( (int) threadIdx.y + l_i < BLOCK_SIZE + MASK_WIDTH - 1) ) {
for(l_j = 0; l_j < MASK_WIDTH; l_j++){
if( ( (int) threadIdx.x + l_j >= 0 ) && ( (int) threadIdx.x + l_j < BLOCK_SIZE + MASK_WIDTH - 1) ) {
if(local[threadIdx.y + l_i][threadIdx.x + l_j] == '1')
amount++;
}
}
}
}

if(source[index] == '1')
amount--;

if(source[index] == '1') {
if(amount < 2 || amount > 3)
goal[index] = '0';
else
goal[index] = '1';
}
else {
if(amount == 3)
goal[index] = '1';
else
goal[index] = '0';
}
}
}