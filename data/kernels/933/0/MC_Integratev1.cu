#include "includes.h"


using namespace std;

#define MAX_N_TERMS 10



__global__ void MC_Integratev1(float* degrees,int dimension,int n_terms,float* I_val,curandState *states, long int seed,int thread_max_iterations)
{
//Get the Global ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

float x;
float I = 0.0;
float f[MAX_N_TERMS];
//float* f =new float[n_terms];

//Initialize the random number generator
curand_init(seed, id, 0, &states[id]);

for (int iter_count=0;iter_count< thread_max_iterations;iter_count++)
{
//Initialize f with the coefficients
for (int term_i=0;term_i<n_terms;term_i++)
{
f[term_i]=degrees[(2+term_i)*dimension];
}

for (int d=1;d<dimension;d++)
{
//Generate a random number in the range of the limits of this dimension
x = curand_uniform (&states[id]);    //x between 0 and 1
//Generate dimension sample based on the limits of the dimension
x = x*(degrees[1*dimension+d]-degrees[0*dimension+d])+degrees[0*dimension+d];
for (int term_i=0;term_i<n_terms;term_i++)
{
//Multiply f of this term by x^(power of this dimension in this term)
f[term_i]*=pow(x,degrees[(2+term_i)*dimension+d]);
}
}
//Add the evaluation to the private summation
for (int term_i=0;term_i<n_terms;term_i++)
{
I+=f[term_i];
}
}
//Add the private summation to the global summation
atomicAdd(I_val,I);

}