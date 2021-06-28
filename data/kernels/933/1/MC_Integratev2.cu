#include "includes.h"


using namespace std;

#define MAX_N_TERMS 10



__global__ void MC_Integratev2(float* degrees_g,int dimension,int n_terms,float* I_val, long int seed,int thread_max_iterations)
{
//Get the global and local ids
int id = blockIdx.x*blockDim.x+threadIdx.x;
int lid=threadIdx.x;
float x;
float I = 0.0;
float f[MAX_N_TERMS];
//float* f =new float[n_terms];

//Dynamically allocate shared memory for 'degrees' and 'I_shared'
extern __shared__ float shared_mem[];
float* I_shared = shared_mem;
I_shared[0]=0;
float* degrees = &shared_mem[1];

//Initialize the local copy of 'degrees' for the shared copy
if(lid<(2+n_terms)*dimension)
{
//copy one element of degrees
degrees[lid]=degrees_g[lid];
}

// Create a state in private memory
curandState state;
//Initialize the random number generator
curand_init(seed,id,0,&state);

//Synchronize all threads to assure that 'degrees' is initialized
__syncthreads();

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
x = curand_uniform (&state);    //x between 0 and 1
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
//Add the private summation to the shared summation
atomicAdd(I_shared,I);
//Synchronize all the threads to assure they all added their private summations to the shared summation
__syncthreads();
//Thread 0 in the block add the shared summation to the global summation
if(lid==0)
{
atomicAdd(I_val,*I_shared);
}


}