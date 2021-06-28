#include "includes.h"



using namespace std;


float *valuesf;
float *weightf;
float maxWf;

float *matchf;
const int fSUMFLAG=0;
const int fKNAPSACKFLAG = 1;

const int fAVGFLAG=2;
const int fMATCHFLAG=3;
const int fINVERSESUMFLAG=4;

const int fMAXIMIZE=-1;
const int fMINIMIZE=1;




__device__ void Crossover(int *chromosome,int size,int start1,int end1,int start2,int end2,int  mid){

for(int i=mid;i<size;i++){
int c1 = start1+mid;
int c2 = start2+mid;
int temp = chromosome[c1];//
//printf("temp =%d and c1 = %d and c2 = %d and ch[c2] = %d\n",temp,c1,c2,chromosome[c2]);

chromosome[c1]=	chromosome[c2];
chromosome[c2]=temp;
}
}
__device__ float generateRandom( curandState* globalState)
{
//int ind = threadIdx.x;
curandState localState = globalState[0];
float RANDOM = curand_uniform( &localState );
globalState[0] = localState;
return RANDOM;
}
__device__ void Crossover(char *chromosome,int size,int start1,int end1,int start2,int end2,int  mid){

for(int i=mid;i<size;i++){
int c1 = start1+mid;
int c2 = start2+mid;
int temp = chromosome[c1];//
//printf("temp =%d and c1 = %d and c2 = %d and ch[c2] = %d\n",temp,c1,c2,chromosome[c2]);

chromosome[c1]=	chromosome[c2];
chromosome[c2]=temp;
}
}
__device__ void Crossover(float *chromosome,int size,int start1,int end1,int start2,int end2,int  mid){

for(int i=mid;i<size;i++){
int c1 = start1+mid;
int c2 = start2+mid;
int temp = chromosome[c1];//
//printf("temp =%d and c1 = %d and c2 = %d and ch[c2] = %d\n",temp,c1,c2,chromosome[c2]);

chromosome[c1]=	chromosome[c2];
chromosome[c2]=temp;
}
}
__device__ float generateRandomf( curandState* globalState)
{
//int ind = threadIdx.x;
curandState localState = globalState[0];
float RANDOM = curand_uniform( &localState );
globalState[0] = localState;
return RANDOM;
}
__global__ void gpuCrossover(float *chromosome,curandState *globalState,int sizeofChromosome,int sizeofPopulation,int Bias,float prob){
int idx = blockIdx.x*blockDim.x+threadIdx.x;
int mid =(int) (generateRandomf(globalState)*sizeofChromosome);//4;// (int) (generateRandom(globalState)*(sizeofChromosome-1));
//printf("MID: %d\n", mid);
idx=idx*2;
int start1,end1;

int start2,end2;
start1 = idx*sizeofChromosome;
end1 = start1+sizeofChromosome;
start2 = end1;
end2 = start2+sizeofChromosome;
if(end2<(sizeofChromosome*sizeofPopulation) )
Crossover(chromosome,sizeofChromosome,start1,end1,start2,end2,mid);
int number = (int) (generateRandomf(globalState)*100);
if(number<(prob*100)){
int j = (int) (generateRandomf(globalState)*((int)sizeofChromosome/4));
for(int k=0;k<j;k++){
int index = (int) (generateRandomf(globalState)*sizeofChromosome);
float a = chromosome[index];// = //(chromosome[index]+1)%2;
if(a==1){
chromosome[index]=0;

}
else{
chromosome[index]=1;
}
}
}

}