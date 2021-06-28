#include "includes.h"

const int CHUNKS = 64;
const int GENERATIONS = 10;


const int CHECK_VALUES_EVERY = 50000;
const int SHOW_ALL_VALUES = 0;
const int SKIP_CUDA_DEVICE = false;


const int ISLANDS_PER_ROW = 4;
const int GENOME_LENGTH=4;
const int BLOCKS_PER_ROW = 4;
const int ISLAND_POPULATION=100;
const int SELECTION_COUNT=80;
const float MUTATION_CHANCE= 0.8;
const int ITEMS_MAX_WEIGHT = 5;
const int ITEMS_MAX_VALUE = 20;
const int ITEMS_MAX = 20;

__global__ void randomizePopulation(curandState_t* states, unsigned char* population ) {
int island_y = blockDim.y * blockIdx.y + threadIdx.y;
int island_x = blockDim.x * blockIdx.x + threadIdx.x;

__shared__ curandState_t randomState;
randomState = states[blockDim.y * blockIdx.y ];

unsigned char * populationRow = &population[island_y * GENOME_LENGTH * ISLAND_POPULATION * ISLANDS_PER_ROW + island_x * GENOME_LENGTH * ISLAND_POPULATION ];

for(int i = 0; i < GENOME_LENGTH * ISLAND_POPULATION; i++) {
populationRow[i] = curand(&randomState) % ITEMS_MAX;
};
}