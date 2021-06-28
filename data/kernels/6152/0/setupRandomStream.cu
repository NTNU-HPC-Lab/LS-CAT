#include "includes.h"
using namespace std;

#define THREADS_PER_BLOCK 32
#define NUM_BLOCKS 32

typedef double HighlyPrecise;

const int GENOME_LENGTH = 14;
const int GENE_MAX = 1;

const float MUTATION_FACTOR = 0.2;
const float CROSSOVER_RATE = 0.6;

const int NUM_EPOCHS = 1000;

struct Chromosome {
HighlyPrecise genes[GENOME_LENGTH];
HighlyPrecise fitnessValue;
};


__global__ void setupRandomStream(unsigned int seed, curandState* states) {
int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
curand_init(seed, threadIndex, 0, &states[threadIndex]);
}