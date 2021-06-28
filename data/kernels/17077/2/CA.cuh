/* Cell Assembly Class Header File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef CA_H
#define CA_H

#include "Synapse.cuh"

// CUDA KERNELS
__global__ void dotP_kernel(float* product, int start, int stop, int* RI, float* data, bool* d_flags);
__global__ void updatePhi_kernel(int n, bool* d_flags, float* d_energy, float* d_fatigue, float theta);
__global__ void updateE_kernel(int n, float* d_energy, int c_decay, float* product, int* CO, int* RI, float* data, bool* d_flags);
__global__ void updateF_kernel(int n, float* d_fatigue, bool* const d_flags, float f_fatigue, float f_recover);

class CA : public Synapse{

private:
    // Default values for CA intiation
    static int d_n_neuron;
    static float d_activity;
    static float d_connectivity;
    static float d_inhibitory;
    static float d_threshold;
    static float d_C[7];
    static int d_available;

protected:
    // Data
    int n_threshold;
    int n_inhibitory;
    int n_excitatory;
    int n_activation;
    bool ignition;

    // Constant Parameters
    float theta; // firing threshold
    float c_decay; // decay constant d
    float f_recover; // recovery constant F^R
    float f_fatigue; // fatigue constant F^C


    float* product;

    // Updates
    void updateFlags(std::vector<bool>& flag_vec,
                     const std::vector<float>& energy_vec,
                     const std::vector<float>& fatigue_vec,
                     const float& theta);

    void updateFlags(int n,
        bool*& h_flags,
        float*& const h_energy,
        float*& const h_fatigue,
        const float& theta);

    void updateE(std::vector<float>& energy_vec,
                 const std::vector<std::vector<float>>& weight_vec,
                 const std::vector<bool>& flag_vec,
                 const int& c_decay);

    void updateE(int n,
        float*& h_energy,
        CSC*& const h_weights,
        bool*& const h_preFlags,
        const int& c_decay);


    void updateF(std::vector<float>& fatigue_vec,
                 const std::vector<bool>& flag_vec,
                 const float& f_fatigue,
                 const float& f_recover);

    void updateF(int n, 
        float*& h_fatigue,
        bool*& const h_flags,
        float& const f_fatigue,
        float& const f_recover);

    //Methods
    float dotP(const std::vector<float>& weights_vec,
               const std::vector<bool>& flags_vec);

    float dotP(const int& start,
        const int& stop,
        int*& const RI,
        float*& const data,
        bool*& const flags);

    

    cudaError_t getDeviceToHostEF(const int& n, float*& h_EF, float*& const d_EF);
    cudaError_t getDeviceToHostFlags(const int& n, bool*& h_flags, bool*& d_flags);
    cudaError_t errorCheckCUDA(bool synchronize = true);

    cudaError_t updatePreGPU();
    cudaError_t updatePostGPU();
    cudaError_t updateE_GPU(dim3 gridSize, dim3 blockSize, bool synchronize = true, bool memCopy = true);
    cudaError_t updateF_GPU(dim3 gridSize, dim3 blockSize, bool synchronize = true, bool memCopy = true);
    cudaError_t updateWeights_GPU(dim3 gridSize, dim3 blockSize, bool synchronize = true, bool memCopy = true);
    cudaError_t updatePhi_GPU(dim3 gridSize, dim3 blockSize, bool synchronize = true, bool memCopy = true);
    //float dotP(const CSC*& h_weights,
    //    const bool*& h_flags);

    //__device__ void dotP_GPU(float* v1, float* v2, float* product, unsigned int n);

public:
    // Constructors - Destructors
    CA(int n = d_n_neuron,
       float activity = d_activity,
       float connectivity = d_connectivity,
       float inhibitory = d_inhibitory,
       float threshold = d_threshold,
       float* C = d_C);
    ~CA();

    void initCADevice();
    void freeCADevice();

    // Running
    void runFor_CPU(int timeStep, int available = d_available);
    void update_CPU();

    void runFor_GPU(int timeStep, int available = d_available);
    void update_GPU();

    

    // GET
    bool getIgnition();

    // Proof of Concept
    static void POC_CPU();
    static void POC_GPU();
};

#endif // CA_H