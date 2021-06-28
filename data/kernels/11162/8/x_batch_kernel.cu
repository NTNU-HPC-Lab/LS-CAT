#include "includes.h"

#define SIZ 20
#define num_inp 4

using namespace std;



typedef struct edge {
int first, second;
} edges;





__global__ void x_batch_kernel(double* X_batch, double * X, int * sample_indices, int size)
{
int i = blockIdx.x;
int j = threadIdx.x;
X_batch[i*size + j] = X[sample_indices[i] * size + j];
}