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

__global__ void init(unsigned int seed, curandState_t* states) {

/* we have to initialize the state */
curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
blockDim.y * blockIdx.y , /* the sequence number should be different for each core (unless you want all
cores to get the same sequence of numbers for some reason - use thread id! */
0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
&states[blockDim.y * blockIdx.y ]);
}