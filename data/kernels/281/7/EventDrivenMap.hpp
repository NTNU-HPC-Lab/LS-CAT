#ifndef EVENTDRIVEMAPHEADERDEF
#define EVENTDRIVEMAPHEADERDEF

#include <cmath>
#include <armadillo>
#include <curand.h>
#include <cassert>
#include "AbstractNonlinearProblem.hpp"
#include "AbstractNonlinearProblemJacobian.hpp"

class EventDrivenMap:
  public AbstractNonlinearProblem
{

  public:

    // Specialised constructor
    EventDrivenMap( const arma::vec* pParameters, unsigned int noReal);

    // Destructor
    ~EventDrivenMap();

    // Right-hand side
    void ComputeF( const arma::vec& u, arma::vec& f);

    // Equation-free stuff
    void SetTimeHorizon( const float T);

    // CUDA stuff
    // Change number of realisations
    void SetNoRealisations( const int noReal);

    void SetNoThreads( const int noThreads);

    // Set variance
    void SetParameterStdDev( const float sigma);

    // Set parameter
    void SetParameters( const unsigned int parId, const float parVal);

    // Reset seed
    void ResetSeed();

    // Set new seed
    void SetNewSeed();

    // Post process data
    void PostProcess();

    // Toggle debug flag
    void SetDebugFlag( const bool val);

    // Structure to store firing times and indices */
    struct __align__(8) firing{
      float time;
      unsigned int index;
    };

  private:

    // Hiding default constructor
    EventDrivenMap();

    // Float vector for temporary storage
    arma::fvec* mpU;
    arma::fvec* mpF;

    // Float vector for parameters
    arma::fvec* mpHost_p;

    // Integration time
    float mFinalTime;

    // threads & blocks
    unsigned int mNoReal;
    unsigned int mNoThreads;
    unsigned int mNoBlocks;
    unsigned int mNoSpikes;

    // CPU variables
    unsigned short *mpHost_lastSpikeInd;

    // GPU variables
    float *mpDev_p;
    float *mpDev_beta;
    float *mpDev_v;
    float *mpDev_s;
    float *mpDev_w;
    float *mpDev_U;
    float *mpDev_lastSpikeTime;
    float *mpDev_crossedSpikeTime;
    unsigned short *mpDev_lastSpikeInd;
    unsigned short *mpDev_crossedSpikeInd;
    unsigned int *mpDev_accept;

    curandGenerator_t mGen; // random number generator
    unsigned long long mSeed; // seed for RNG
    float mParStdDev;

    // Functions to do lifting
    void initialSpikeInd( const arma::vec& U);

    void ZtoU( const arma::vec& Z, arma::vec& U);

    void UtoZ( const arma::vec *U, arma::vec *Z);

    void BuildCouplingKernel();

    // For debugging purposes
    bool mDebugFlag;

    void SaveInitialSpikeInd();

    void SaveLift();

    void SaveEvolve();

    void SaveRestrict();

    void SaveAveraged();
};

__global__ void LiftKernel( float *s, float *v, const float *par, const float *U,
    const unsigned int noReal);

// Functions to find spike time
__device__ float fun( float t, float v, float s, float beta);

__device__ float dfun( float t, float v, float s, float beta);

__device__ float eventTime( float v0, float s0, float beta);

// evolution
__global__ void EvolveKernel( float *v, float *s, const float *beta,
    const float *w, const float finalTime, unsigned short *global_lastSpikeInd,
    float *global_lastFiringTime, unsigned short *global_crossedSpikeInd,
    float *global_crossedFiringTime, unsigned int *global_accept, unsigned int noReal);

// restriction
__global__ void RestrictKernel( float *global_lastSpikeTime,
                                const unsigned short *global_lastSpikeInd,
                                const float *global_crossedSpikeTime,
                                const unsigned short *global_crossedSpikeInd,
                                const float finalTime,
                                const unsigned int noReal);

// count number of active realisations
__global__ void CountRealisationsKernel( unsigned int *accept, const unsigned int noReal);

// averaging functions
__global__ void realisationReductionKernelBlocks( float *dev_V,
                                                  const float *dev_U,
                                                  const unsigned int noReal,
                                                  const unsigned int *accept);

// helper functions
__global__ void initialSpikeIndCopyKernel( unsigned short* pLastSpikeInd, const unsigned int noReal);

void circshift( float *w, int shift, unsigned int noThreads);
__device__ struct EventDrivenMap::firing warpReduceMin( struct EventDrivenMap::firing val);
__device__ struct EventDrivenMap::firing blockReduceMin( struct EventDrivenMap::firing val);
__device__ float warpReduceSum ( float val);
__device__ float blockReduceSum( float val);
__device__ int warpReduceSumInt ( int val);
__device__ int blockReduceSumInt( int val);

void SaveData( int npts, float *x, char *filename);

#endif
