#pragma once

/**

Compile Time Cuda Constants

Thread per Block etc..

*/

#ifdef METU_SHARED_GPULIST
#define METU_SHARED_TRACER_ENTRY_POINT __declspec(dllexport)
#else
#define METU_SHARED_TRACER_ENTRY_POINT __declspec(dllimport)
#endif

#include <cuda.h>
#include <set>
#include <array>
#include <vector>

#include "RayLib/Vector.h"
#include "RayLib/Error.h"

// Except first generation this did not change having this compile time constant is a bliss
static constexpr unsigned int WarpSize = 32;

// Thread Per Block Constants
static constexpr unsigned int BlocksPerSM = 32;
static constexpr unsigned int StaticThreadPerBlock1D = 256;
static constexpr unsigned int StaticThreadPerBlock2D_X = 16;
static constexpr unsigned int StaticThreadPerBlock2D_Y = 16;
static constexpr Vector2ui StaticThreadPerBlock2D = Vector2ui(StaticThreadPerBlock2D_X,
                                                              StaticThreadPerBlock2D_Y);

struct CudaError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            OLD_DRIVER,
            NO_DEVICE,
            // End
            END
        };

    private:
        Type        type;

    public:
        // Constructors & Destructor
                    CudaError(Type);
                    ~CudaError() = default;

         operator   Type() const;
         operator   std::string() const override;
};

inline CudaError::CudaError(CudaError::Type t)
    : type(t)
{}

inline CudaError::operator Type() const
{
    return type;
}

inline CudaError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        "Driver is not up-to-date",
        "No cuda capable device is found"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(CudaError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}

class CudaGPU
{
    public:
        enum GPUTier
        {
            GPU_UNSUPPORTED,
            GPU_KEPLER,
            GPU_MAXWELL,
            GPU_PASCAL,
            GPU_TURING_VOLTA
        };

        static GPUTier              DetermineGPUTier(cudaDeviceProp);

    private:
        struct WorkGroup
        {
            static constexpr size_t                 MAX_STREAMS = 64;
            std::array<cudaEvent_t, MAX_STREAMS>    events;
            std::array<cudaStream_t, MAX_STREAMS>   works;
            cudaEvent_t                             mainEvent;
            mutable int                             currentIndex;
            int                                     totalStreams;

            // Constructors
                                            WorkGroup();
                                            WorkGroup(int deviceId, int streamCount);
                                            WorkGroup(const WorkGroup&) = delete;
                                            WorkGroup(WorkGroup&&);
            WorkGroup&                      operator=(const WorkGroup&) = delete;
            WorkGroup&                      operator=(WorkGroup&&);
                                            ~WorkGroup();

            cudaStream_t                    UseGroup() const;
            void                            WaitAllStreams() const;
            void                            WaitMainStream() const;
        };


    private:    
        int                     deviceId;
        cudaDeviceProp          props;
        GPUTier                 tier;
        WorkGroup               workList;

        uint32_t                DetermineGridStrideBlock(uint32_t sharedMemSize,
                                                         uint32_t threadCount,
                                                         size_t workCount,
                                                         void* func) const;

    protected:
    public:
        // Constrctors & Destructor
        explicit                CudaGPU(int deviceId);
                                CudaGPU(const CudaGPU&) = delete;
                                CudaGPU(CudaGPU&&) = default;
        CudaGPU&                operator=(const CudaGPU&) = delete;
        CudaGPU&                operator=(CudaGPU&&) = default;
                                ~CudaGPU() = default;
        //
        int                     DeviceId() const;
        std::string             Name() const;
        double                  TotalMemoryMB() const;
        double                  TotalMemoryGB() const;
        GPUTier                 Tier() const;

        size_t                  TotalMemory() const;
        Vector2i                MaxTexture2DSize() const;

        uint32_t                SMCount() const;
        uint32_t                RecommendedBlockCountPerSM(void* kernkernelFuncelPtr,
                                                           uint32_t threadsPerBlock = StaticThreadPerBlock1D,
                                                           uint32_t sharedMemSize = 0) const;
        cudaStream_t            DetermineStream(uint32_t requiredSMCount) const;
        void                    WaitAllStreams() const;
        void                    WaitMainStream() const;

        bool                    operator<(const CudaGPU&) const;

                // Classic GPU Calls
        // Create just enough blocks according to work size
        template<class Function, class... Args>
        __host__ void           KC_X(uint32_t sharedMemSize,
                                     cudaStream_t stream,
                                     size_t workCount,
                                     //
                                     Function&& f, Args&&...) const;
        template<class Function, class... Args>
        __host__ void           KC_XY(uint32_t sharedMemSize,
                                      cudaStream_t stream,
                                      size_t workCount,
                                      //
                                      Function&& f, Args&&...) const;

        // Grid-Stride Kernels
        // Convenience Functions For Kernel Call
        // Simple full GPU utilizing calls over a stream
        template<class Function, class... Args>
        __host__ void           GridStrideKC_X(uint32_t sharedMemSize,
                                               cudaStream_t stream,
                                               size_t workCount,
                                               //
                                               Function&&, Args&&...) const;

        template<class Function, class... Args>
        __host__ void           GridStrideKC_XY(uint32_t sharedMemSize,
                                                cudaStream_t stream,
                                                size_t workCount,
                                                //
                                                Function&&, Args&&...) const;

        // Smart GPU Calls
        // Automatic stream split
        // Only for grid strided kernels, and for specific GPU
        // Material calls require difrrent GPUs (texture sharing)
        // TODO:
        template<class Function, class... Args>
        __host__ void           AsyncGridStrideKC_X(uint32_t sharedMemSize,
                                                     size_t workCount,
                                                     //
                                                     Function&&, Args&&...) const;

        template<class Function, class... Args>
        __host__ void           AsyncGridStrideKC_XY(uint32_t sharedMemSize,
                                                     size_t workCount,
                                                     //
                                                     Function&&, Args&&...) const;
};

// Verbosity
using GPUList = std::set<CudaGPU>;

class CudaSystem
{
    private:
        GPUList                systemGPUs;
        //CudaError                           systemStatus;

    protected:
    public:
        // Constructors & Destructor
                                    CudaSystem() = default;
                                    CudaSystem(const CudaSystem&) = delete;

        CudaError                   Initialize();

        // Multi-Device Splittable Smart GPU Calls
        // Automatic device split and stream split on devices
        const std::vector<size_t>   GridStrideMultiGPUSplit(size_t workCount,
                                                            uint32_t threadCount,
                                                            uint32_t sharedMemSize,
                                                            void* f) const;

        // Misc
        const GPUList&              GPUList() const;

        // Device Synchronization
        void                        SyncGPUMainStreamAll() const;
        void                        SyncGPUMainStream(const CudaGPU& deviceId) const;

        void                        SyncGPUAll() const;
        void                        SyncGPU(const CudaGPU& deviceId) const;
};

template<class Function, class... Args>
__host__
void CudaGPU::KC_X(uint32_t sharedMemSize,
                   cudaStream_t stream,
                   size_t workCount,
                   //
                   Function&& f, Args&&... args) const
{
    CUDA_CHECK(cudaSetDevice(deviceId));
    uint32_t blockCount = static_cast<uint32_t>((workCount + (StaticThreadPerBlock1D - 1)) / StaticThreadPerBlock1D);
    uint32_t blockSize = StaticThreadPerBlock1D;
    f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
void CudaGPU::KC_XY(uint32_t sharedMemSize,
                    cudaStream_t stream,
                    size_t workCount,
                    //
                    Function&& f, Args&&... args) const
{
    CUDA_CHECK(cudaSetDevice(deviceId));
    size_t linearThreadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
    size_t blockCount = (workCount + (linearThreadCount - 1)) / StaticThreadPerBlock1D;
    uint32_t blockSize = StaticThreadPerBlock1D;
    f<<<blockCount, blockSize, sharedMemSize, stream>>> (args...);
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
inline void CudaGPU::GridStrideKC_X(uint32_t sharedMemSize,
                                    cudaStream_t stream,
                                    size_t workCount,
                                    //
                                    Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock1D;
    uint32_t blockCount = DetermineGridStrideBlock(sharedMemSize,
                                                   threadCount, workCount, &f);

    // Full potential GPU Call
    CUDA_CHECK(cudaSetDevice(deviceId));
    uint32_t blockSize = StaticThreadPerBlock1D;
    f<<<blockCount, blockSize, sharedMemSize, stream>>> (args...);
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__
inline void CudaGPU::GridStrideKC_XY(uint32_t sharedMemSize,
                                     cudaStream_t stream,
                                     size_t workCount,
                                     //
                                     Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
    uint32_t blockCount = DetermineGridStrideBlock(sharedMemSize,
                                                   threadCount, workCount, &f);

    CUDA_CHECK(cudaSetDevice(deviceId));
    dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);
    f<<<blockCount, blockSize, sharedMemSize, stream>>>(args...);
    CUDA_KERNEL_CHECK();
}


template<class Function, class... Args>
__host__ void CudaGPU::AsyncGridStrideKC_X(uint32_t sharedMemSize,
                                           size_t workCount,
                                           //
                                           Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock1D;
    uint32_t requiredSMCount = DetermineGridStrideBlock(sharedMemSize,
                                                        threadCount, workCount, &f);
    cudaStream_t stream = DetermineStream(requiredSMCount);

    CUDA_CHECK(cudaSetDevice(deviceId));
    uint32_t blockSize = StaticThreadPerBlock1D;
    f<<<requiredSMCount, blockSize, sharedMemSize, stream>>>(args...);
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
__host__ void CudaGPU::AsyncGridStrideKC_XY(uint32_t sharedMemSize,
                                            size_t workCount,
                                            //
                                            Function&& f, Args&&... args) const
{
    const size_t threadCount = StaticThreadPerBlock2D[0] * StaticThreadPerBlock2D[1];
    uint32_t requiredSMCount = DetermineGridStrideBlock(sharedMemSize,
                                                        thread, workCount, &f);
    cudaStream_t stream = GPUList()[gpuIndex].DetermineStream(requiredSMCount);

    CUDA_CHECK(cudaSetDevice(deviceId));
    dim3 blockSize = dim3(StaticThreadPerBlock2D[0], StaticThreadPerBlock2D[1]);
    f<<<requiredSMCount, blockSize, sharedMemSize, stream>>>(args...);
    CUDA_KERNEL_CHECK();
}

inline const GPUList& CudaSystem::GPUList() const
{
    return systemGPUs;
}

inline void CudaSystem::SyncGPUMainStreamAll() const
{
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    for(const auto& gpu : systemGPUs)
    {
        gpu.WaitMainStream();
    }
    CUDA_CHECK(cudaSetDevice(currentDevice));
}

inline void CudaSystem::SyncGPUMainStream(const CudaGPU& gpu) const
{
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    auto i = systemGPUs.end();
    if((i = systemGPUs.find(gpu)) != systemGPUs.end())
    {
        i->WaitMainStream();
    }
    CUDA_CHECK(cudaSetDevice(currentDevice));
}

inline void CudaSystem::SyncGPUAll() const
{
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    for(const auto& gpu : systemGPUs)
    {
        gpu.WaitAllStreams();
        gpu.WaitMainStream();
        break;
    }
    CUDA_CHECK(cudaSetDevice(currentDevice));
}

inline void CudaSystem::SyncGPU(const CudaGPU& gpu) const
{
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    auto i = systemGPUs.end();
    if((i = systemGPUs.find(gpu)) != systemGPUs.end())
    {
        i->WaitAllStreams();
        i->WaitMainStream();
    }
    CUDA_CHECK(cudaSetDevice(currentDevice));
}