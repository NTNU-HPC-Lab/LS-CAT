#include "includes.h"

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define SHARED_BLOCK_DIM 32
#define CHUNK_SIZE 512
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
__global__ void show(){
//printf("AAAAAAAAAAAAA\n");
/*for(int i=0;i<global_width*global_height;i++){
printf("%d\n",next_cu[i]);
}*/
}