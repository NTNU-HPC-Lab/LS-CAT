/**
* A common header containing forward declarations and type definitions dependent on the cuda
* runtime
*/

#ifndef CULINALG_HEADER_CUHEADER
#define CULINALG_HEADER_CUHEADER

#include<string>


/**
 * The number of threads in a thread block. The value 64 is motivated by the attempt to have two
 * warps in a thread block. Any fewer, and the scheduler inside an SMP may potentially idle if only
 * a single block is loaded. Any more, and systems with large number of SMPs may end up with SMPs
 * with no load.
 */
#define CULINALG_BLOCK_SIZE 64


namespace clg
{
    /**
     * A class representing a large CUDA object with memory beinged mirrored between host and device.
     * Used to store the data for vectors and matrices. Template parameter T is expected to be iether
     * float or double.
     */
    struct CuData
    {
        /**
         * Pointers to data for host and device. Should either be both nullptr, or both refer to valid memory
         */ 
        void* host_data; 
        void* device_data;
        /**
         * The following flag is true if the host_data contains the correct data for the vector,
         * otherwise the device_data contains the correct data.
         */
        bool host_data_synced;

        // TODO Use for async
        ///**
        // * Events reading to or writing from this object. Both cannot be nonempty, as reading to and
        // * writing from the same object simultaneously should not happen
        // */
        //std::queue<CudaEvent_t> reader_events;
        //CudaEvent_t writer_events;

        /**
         * Reset this CuData to a valid state referring to no data
         */
        void reset();
        /**
         * Move data from given CuData to this CuData. The result of this will leave both CuData
         * objects pointing to the same data.
         */
        void move_from(const CuData& other);
        /**
         * Synchronizes the memory segment starting at where this `CuData` refers to and extending for
         * `size` bytes so that the host side memory holds the correct data. Strong exception
         * guarantee.
         */
        void memsync_host(size_t size);
        /**
         * Synchronizes the memory segment starting at where this `CuData` refers to and extending for
         * `size` bytes so that the device side memory holds the correct data. Strong exception
         * guarantee.
         */
        void memsync_device(size_t size);
    };

    /**
     * A function that wraps a CUDA call to check for errors and if one occurs it throws the error
     * of the passed template arguement type. Template arguement is expected to be a valid error
     * defined in headers/culinalg-exceptions.hpp. Note that this function itself does not provide
     * any exception guarantee, and is intended to simply be syntactic sugar for the often repeated
     * if-throw-get-message pattern for wrapping CudaErrors in exceptions.
     */
    template<class E> inline void wrapCudaError(const cudaError_t& err)
    {
        if(err != cudaSuccess)
            throw E("CUDA Error: " + std::string(cudaGetErrorName(err)) + ": " +
                    std::string(cudaGetErrorString(err)));
    }


    /**
     * Copies data from the second CuData arguement into the first. Copies as many bytes as the 
     * third arguement. Throws CopyFailedException, either due to internal CUDA errors, or because 
     * the CuDatas do not point to any data. Both source and destination CuDatas remain valid, that 
     * is, they continue to point to some date mirrored on host and device, but a failed copy may 
     * corrupt the data itself. This essentially provides strong exception guarantee with respect to
     * validity of CuData invariants.
     */
    void copyCuData(const CuData& dst, const CuData& src, size_t count);
    


    /**
     * A thin wrapper class for CudaEvent_t.
     */
    // TODO complete
    //class CuEvent
    //{
    //    public:
    //        /**
    //         * Constructs a new cuda event, calls cudaEventCreate()
    //         */
    //        CuEvent() { cudaEventCreate(&event); }
    //        /**
    //         * Destroys an event, calls 

    //    private:
    //        CudaEvent_t event;
    //};
}

#endif
