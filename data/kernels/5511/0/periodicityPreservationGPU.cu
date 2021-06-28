#include "includes.h"

#define BUFSIZE 64
#define BLOCK_SIZE 16

// Perdiodicty Preservation retains our periodicity
// Runs on CPU
__global__ void periodicityPreservationGPU(int N, char *cells)
{
int i;
//rows
for (i = 1; i <= N; ++i)
{
//Copy first real row to bottom extra row
cells[(N+2)*(N+1)+i] = cells[(N+2)+i];
//Copy last real row to top extra row
cells[i] = cells[(N+2)*N + i];
}
//cols
for (i = 0; i <= N+1; ++i)
{
//Copy first real column to right last extra column
cells[i*(N+2)+N+1] = cells[i*(N+2)+1];
//Copy last real column to left last extra column
cells[i*(N+2)] = cells[i*(N+2) + N];
}
}