#include "includes.h"
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
__global__ void gpuCrossover(int *chromosome,curandState *globalState,int sizeofChromosome,int sizeofPopulation,int Bias,float prob){
int idx = blockIdx.x*blockDim.x+threadIdx.x;
int mid =(int) (generateRandom(globalState)*sizeofChromosome);//4;// (int) (generateRandom(globalState)*(sizeofChromosome-1));
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
int number = (int) (generateRandom(globalState)*100);
if(number<(prob*100)){
int j = (int) (generateRandom(globalState)*((int)sizeofChromosome/4));
for(int k=0;k<j;k++){
int index = (int) (generateRandom(globalState)*sizeofChromosome);
int a = chromosome[index];// = //(chromosome[index]+1)%2;
if(a==1){
chromosome[index]=0;

}
else{
chromosome[index]=1;
}
}
}

}