/* Synapse Class Header File
 *
 * 200619
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef SYNAPSE_H
#define SYNAPSE_H

#include "FLIF.cuh"

 // CUDA KERNELS
__global__ void updateWeights_kernel(const int pre_size, 
    bool* const d_preFlags,
    const int post_size,
    bool* const d_postFlags,
    const float alpha,
    const float w_average,
    const float w_current,
    int* const CO,
    int* const RI,
    float* data);

class Synapse : public FLIF{
private:

protected:
    float connectivity;
    float inhibitory;
    float alpha; // learning rate
    float w_average; // constant representing average total synaptic strength of the pre-synaptic neuron.
    float w_current; // current total synaptic strength

    // Synapse ==== FLIF
    std::vector<bool> pre_flags;
    std::vector<bool> post_flags;

    // Host Flags
    int preSize;
    int postSize;

    bool* h_preFlags;
    bool* h_postFlags;

    // Device Flags
    bool* d_preFlags; 
    bool* d_postFlags;

    std::vector<FLIF*> incomingList;
    std::vector<FLIF*> outgoingList;
    

    // Init === FLIF
    void initWeights(int in, int out, float connectivity, float inhibitory,
        std::vector<std::vector<float>>& weight_vec);

    // Host Inits
    void initWeights(int in, int out, float connectivity, float inhibitory,
        CSC*& h_weights);

    void COOToCSC(CSC*& target, const std::vector<COO>& source, int row, int col);
    void CSCToDense(std::vector<std::vector<float>>& target,  CSC*& const source);
    float getDataCSC(CSC*& target, int i, int j);
    void setDataCSC(CSC*& target, int i, int j, const float& data);
    CSC* initCSC(int rowSize, int columnSize, int nonzeros);
    void deleteCSC(CSC*& target);

    // Device Inits
    cudaError_t initCSCDevice(CSC*& d_CSC, CSC*& const h_CSC, bool allocHost=true, bool alloc = true);
    cudaError_t freeCSCDevice(CSC*& d_CSC);

    // Decvice to host
    cudaError_t getDeviceToHostCSC(CSC*& h_CSC, CSC*& const d_CSC);

    // Update
    void updateWeights(std::vector<std::vector<float>>& weight_vec,
        const std::vector<bool>& pre_vec,
        const std::vector<bool>& post_vec,
        const float& alpha,
        const float& w_average,
        const float& w_current);

    void updateWeights(CSC*& h_weights,
                    const int& preSize,
                    bool*& const h_preFlags,
                    const int& postSize,
                    bool*& const h_postFlags,
                    const float& alpha,
                    const float& w_average,
                    const float& w_current);

    //void updateWeightsGPU(CSC*& d_weights,
    //    const bool* d_preFlags,
    //    const bool* d_postFlags,
    //    const float& alpha,
    //    const float& w_average,
    //    const float& w_current);

    void updatePre(std::vector<bool>& pre_synaptic_flags,
        const std::vector<FLIF*>& incoming);

    void updatePost(std::vector<bool>& post_synaptic_flags,
        const std::vector<FLIF*>& outgoing);

    // HOST
    void updatePre(bool*& h_preFlags, 
        int& preSize,
        const std::vector<FLIF*>& incoming);

    void updatePost(bool*& const h_postFlags,
        int& postSize,
        const std::vector<FLIF*>& outgoing);

    // Connect
    void addIncomingWeights(std::vector<std::vector<float>>& resting,
        const std::vector<std::vector<float>>& in);
    void addOutgoingWeights(std::vector<std::vector<float>>& resting,
        const std::vector<std::vector<float>>& out);

    void addIncomingWeights(CSC*& resting, CSC*& const in);
    void addOutgoingWeights(CSC*& resting, CSC*& const out);

public:
    Synapse();
    ~Synapse();
    
    // Connecting
    void connectIn(FLIF* incoming,
        float strength,
        float inhibitory);
    void connectOut(FLIF* outgoing,
        float strength,
        float inhibitory);
    static void connect(Synapse* pre_synaptic, float pre_strength, float pre_inhibitory,
                        Synapse* post_synaptic, float post_strength, float post_inhibitory);

    static void connect_GPU(Synapse* pre_synaptic, float pre_strength, float pre_inhibitory, Synapse* post_synaptic, float post_strength, float post_inhibitory);

    void connectRestore_GPU();

    static void POC();



    

};

#endif // SYNAPSE_H