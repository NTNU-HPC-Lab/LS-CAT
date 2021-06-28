#include "includes.h"


#define DATA float
#define BOOL int
#define MAX_ERR (float)1e-5
#define MAX_EPOCHS 3

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

//Grid features
//Leggere 15 febbraio del diario (passo 1 del feedforward, considerazioni)

#define OPTIMUM_BLOCK_NUM 4 //In vista della concorrenza dei kernels
#define BLOCK_SIDE	16

#define OPTIMUM_BLOCK_NUM_FIRST_LAYER 2
#define BLOCK_SIDE_FIRST_LAYER 32

/*Struct Grid Settings*/

typedef struct grid_settings {
int grid[3];
int block[3];
}grid_settings;

grid_settings gs = { { OPTIMUM_BLOCK_NUM_FIRST_LAYER, OPTIMUM_BLOCK_NUM, OPTIMUM_BLOCK_NUM },{ BLOCK_SIDE_FIRST_LAYER,BLOCK_SIDE,BLOCK_SIDE } };

//Network features

#define NEURO_INPUT 784 //#neuroni dell'input layer
#define NEURO_H_0	56	//#neuroni del primo hidden layer
#define NEURO_H_1	28	//#neuroni del secondo hidden layer
#define NEURO_OUTPUT 10 //#neuroni dell'output layer
#define TOTAL_PATT	60000 //#patterns totali
#define NUM_HIDDEN 2 //#hidden layers
#define TOTAL_LAYER 4 //#di layers

//Streams Settings
#define NSTREAMS 3

//Texture reference (FOR TARGET MATRIX)
texture<DATA, 2, cudaReadModeElementType> texreference_target;

//Constant memory (read by all the threads)
__constant__ DATA alpha_const[1];
__constant__ DATA eta_const[1];

/*UTILITIES*/

__global__ void updateWeightBiasMat(DATA *delta_weightbias, DATA *weight, int rows, int cols) {

int dest_x = blockIdx.x*blockDim.x + threadIdx.x;
int dest_y = blockIdx.y*blockDim.y + threadIdx.y;

if (dest_x < cols && dest_y < rows) {
DATA derivative = delta_weightbias[dest_y*cols + dest_x];
weight[dest_y*cols + dest_x] += derivative;
delta_weightbias[dest_y*cols + dest_x] *= alpha_const[0];
}
}