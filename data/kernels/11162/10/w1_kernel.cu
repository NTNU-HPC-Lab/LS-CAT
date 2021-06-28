#include "includes.h"

#define SIZ 20
#define num_inp 4

using namespace std;



typedef struct edge {
int first, second;
} edges;





__global__ void w1_kernel(double * grads_W1, double * W1, double learning_rate, int size)
{
int i = blockIdx.x;
int j = threadIdx.x;
W1[i*size + j] += (-learning_rate * grads_W1[i*size + j]);
}