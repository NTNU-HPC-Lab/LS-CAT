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


extern const int cSUMFLAG;
extern const int cKNAPSACKFLAG;

extern const int cAVGFLAG;
extern const int cMATCHFLAG;
extern const int cINVERSESUMFLAG;

extern const int cMAXIMIZE;
extern const int cMINIMIZE;

__device__ float getKnapsackFitnessc(char *,int,float*,float*,float,int,int);
__device__ float getAvgCountc(char*,int,int,int);
__device__ float getSumc(char*,int,int,int);
__device__ float getMatchc(char*,char*,int,int,int);
__device__ void Crossoverc(char*,int,int,int,int,int,int);


class EABase
{
	public:
		virtual void initializePopulation()=0;
		virtual void getStatistics()=0;
		
};
/*=========================================BINARY GENOME=================================================*/

class CharChromosome1D
{
	public:

		char *chromosome;
		int sizeofChromosome;
		float fitnessValue;
		CharChromosome1D(int size){
			sizeofChromosome=size;
			chromosome = (char*) malloc(size*sizeof(char));
		}	
		void initializeChromosome(int size){
			sizeofChromosome=size;
			chromosome = (char*) malloc(size*sizeof(char));	
			fitnessValue=0.0;		
		}
		CharChromosome1D(){

		}
};

class EAChar: public EABase
{
	private:
		int id;
		char *chromosome;
		int *indices;
		int *cudaIndices; 
		
		int fitnessFlag;
		int minmaxflag; 
		int popSize;
		float mutationProbability=0.4;
		char *randomRange;
		CharChromosome1D *population;
		char **Cudapopulation1D;
		char *Cudapopulation;
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
		EAChar(int,int,char*);// (sizeofChromosomoe,sizeofpopulation,range[0,1])
		EAChar(int,int,char**);// (sizeofChromosome,sizeofpopulation,population)
		void initializePopulation(){
			init();

		}
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

		void setParamKnapSack(float *kvalues,float *kweight,int chromosomeSize,float maxWeight);
		void setMatchParameter(char *kvalues,int chromosomeSize);

		int getChoromosomeSize();
		int getPopulationSize();
		void printpopulation();
		~EAChar(){
			cudaFree(Cudapopulation);
		}

		

};
