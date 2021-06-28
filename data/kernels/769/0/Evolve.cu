#include "includes.h"



#define BLOCK_SIZE 16




__global__ void Evolve(bool* field, float* scores, double b, int size, bool* next_field)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int memberIndex;

// Score
if (col >= size || row >= size)
return;

//printf("(%i, %i)\n", col, row);

float score = 0;

for (int i = -1; i <= 1; i++) //Row
{
for (int j = -1; j <= 1; j++) //Col
{
memberIndex = (col + i + size) % size + size * ((row + j + size) % size);

if (field[memberIndex] == true)
score++;
}
}

if (!field[row*size + col])
scores[row*size + col] = score * b;
else
scores[row*size + col] = score;


__syncthreads();


// Strategy
int bestStrategyIndex = row*size + col;

for (int i = -1; i <= 1; i++) //Row
{
for (int j = -1; j <= 1; j++) //Col
{
memberIndex = (col + i + size) % size + size * ((row + j + size) % size);

if (scores[bestStrategyIndex] < scores[memberIndex])
{
bestStrategyIndex = memberIndex;
}
}
}

next_field[row*size + col] = field[bestStrategyIndex];

__syncthreads();
}