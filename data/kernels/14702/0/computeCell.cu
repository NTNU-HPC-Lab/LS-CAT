#include "includes.h"
__device__ void applyRule(char* left, char* middle, char* right, char* res){
char a = *left;
char b = *middle;
char c = *right;

if(a == 0 && b == 0 && c == 0){
*res = 0;
}else if(a == 0 && b == 0 && c == 1){
*res = 1;
}else if(a == 0 && b == 1 && c == 0){
*res = 1;
}else if(a == 0 && b == 1 && c == 1){
*res = 1;
}else if(a == 1 && b == 0 && c == 0){
*res = 0;
}else if(a == 1 && b == 0 && c == 1){
*res = 1;
}else if(a == 1 && b == 1 && c == 0){
*res = 1;
}else if(a == 1 && b == 1 && c == 1){
*res = 0;
}
}
__global__ void computeCell(char* cellData, unsigned int* width, unsigned int* height)
{
int y = 0;
int x = 0;

/*
printf("width : %d\n", *width);
printf("height : %d\n", *height);
printf("dimblock : %d\n", blockDim.x);
printf("threadid : %d\n", threadIdx.x);
*/

for(y=1; y < (*height); ++y){
for(x=threadIdx.x; x < (*width); x += blockDim.x){
char left = ((x-1)>=0) ? cellData[(x-1) + (y-1) * (*width)] : 0;
char middle = cellData[x + (y-1) * (*width)];
char right = ((x+1) < *width) ? cellData[(x+1) + (y-1) * (*width)] : 0;

//printf("left   : (%d, %d) => %d : %d\n", x-1, y-1, (x-1) + (y-1) * (*width), left);
//printf("middle : (%d, %d) => %d : %d\n", x, y-1, (x) + (y-1) * (*width), middle);
//printf("right  : (%d, %d) => %d : %d\n", x+1, y-1, (x+1) + (y-1) * (*width), right);
applyRule(&left, &middle, &right, &cellData[x + y * (*width)]);
//printf("res    : (%d, %d) => %d : %d\n", x, y, x + y * (*width), cellData[x+y*(*width)]);

//printf("-----------\n");
}
__syncthreads();
}

/*
for(y=0; y < *height; ++y){
for(x=0; x < *width; ++x){
printf("(%d, %d) = %d\n", x, y, cellData[x+y*(*width)]);
}
}
*/
}