#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <ctime>
#include <algorithm>  // For time()
#include <cstdlib>
#include <chrono>
#include <unistd.h>

#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>


extern const int SUMFLAG;
extern const int KNAPSACKFLAG;

extern const int AVGFLAG;
extern const int MATCHFLAG;
extern const int INVERSESUMFLAG;

extern const int MAXIMIZE;
extern const int MINIMIZE;

__device__ float getKnapsackFitness(int *,int,float*,float*,float,int,int);
__device__ float getAvgCount(int*,int,int,int);
__device__ float getSum(int*,int,int,int);
__device__ float getMatch(int*,int*,int,int,int);
__device__ void Crossover(int*,int,int,int,int,int,int);

class EABaseB
{
	public:
		virtual void initializePopulation()=0;
		virtual void getStatistics()=0;
		
};
/*=========================================BINARY GENOME=================================================*/

class BinaryChromosome1D
{
	public:

		int *chromosome;
		int sizeofChromosome;
		float fitnessValue;
		BinaryChromosome1D(int size){
			sizeofChromosome=size;
			chromosome = (int*) malloc(size*sizeof(int));
		}	
		void initializeChromosome(int size){
			sizeofChromosome=size;
			chromosome = (int*) malloc(size*sizeof(int));	
			fitnessValue=0.0;		
		}
		BinaryChromosome1D(){

		}
};

class EABinary: public EABaseB
{
	private:
		int id;
		int *chromosome;
		int *indices;
		int *cudaIndices; 
		
		int fitnessFlag;
		int minmaxflag; 
		int popSize;
		float mutationProbability=0.4;
		int *randomRange;
		BinaryChromosome1D *population;
		int **Cudapopulation1D;
		int *Cudapopulation;
		int chromosomeSize;
		int populationSize;
		curandState* devStates;
		dim3 threads;
		dim3 blocks;

		//Statistics Variables
		double crossoverMutationTime=0.0;
		double fitnessCalculationTime=0.0;
		double sortingpopulationTime=0.0;
		double totalKernelTime=0.0;
		double totalMemoryTransferTime=0.0;
		double totalGpuTime=0.0;
		double totalCPUTime=0.0;

		


		
	public:

		void setFitnessFlag(int,int);		
		void shuffle(int);
		void evolve();
		void sortpop();
		float fitness(int);
		void init();
		EABinary(int,int,int*);// (sizeofChromosomoe,sizeofpopulation,range[0,1])
		EABinary(int,int,int**);// (sizeofChromosome,sizeofpopulation,population)
		void initializePopulation(){
			init();

		}
		void setParamKnapSack(float *kvalues,float *kweight,int chromosomeSize,float maxWeight);
		void setMatchParameter(int *kvalues,int chromosomeSize);

		void doCrossOver(int);
		void doMutation(int);
		void getStatistics(){
			printf("\n======================================================================================================\n");
			printf("\n===================================STATISTICS=========================================================\n");
			printf("\n======================================================================================================\n");
			printf("Total kernel Execution time : %f\n",totalKernelTime);
			printf("Total Memory tranfer time : %f\n", totalMemoryTransferTime);
			printf("Total Fitness calculation time : %f\n",fitnessCalculationTime);
		}
		int getChoromosomeSize();
		int getPopulationSize();
		void printpopulation();
		~EABinary(){
			cudaFree(Cudapopulation);
		}

		

};
