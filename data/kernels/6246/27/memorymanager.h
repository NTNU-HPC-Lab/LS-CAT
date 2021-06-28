/**
 * @file memorymanager.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-07
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "magmadnn/types.h"

namespace magmadnn {

template <typename T>
class MemoryManager {
   public:
    /** MemoryManager class to keep track of a memory address across devices.
     *  @param size the size of the memory to allocate/manage
     *  @param mem_type what memory type will this data belong to
     *  @param device_id what device will the data reside on (preferred if mem_type is CUDA_MANAGED)
     */
    MemoryManager(unsigned int size, memory_t mem_type, device_t device_id);

    /** Copy Constructor
     * @param that
     */
    MemoryManager(const MemoryManager& that);

    /** Copy assignment operator.
     * @param that
     * @return MemoryManager&
     */
    MemoryManager& operator=(const MemoryManager& that);

    /** Destroys the memory manager object and releases all its data.
     */
    ~MemoryManager();

    /** Copies the data from src memory manager into the pointer here. Asserts that
     *  src and this have the same size.
     *  @param src the memorymanager to copy data from
     *  @return the error code (0 - no error, 1 - src ptr not allocated)
     */
    magmadnn_error_t copy_from(const MemoryManager<T>& src, unsigned int begin_idx, unsigned int size);

    /** Copies the data from src memory manager into the pointer here. Asserts that
     *  src and this have the same size.
     *  @param src the memorymanager to copy data from
     *  @return the error code (0 - no error, 1 - src ptr not allocated)
     */
    magmadnn_error_t copy_from(const MemoryManager<T>& src, unsigned int size);

    /** Copies the data from src memory manager into the pointer here. Asserts that
     *  src and this have the same size.
     *  @param src the memorymanager to copy data from
     *  @return the error code (0 - no error, 1 - src ptr not allocated)
     */
    magmadnn_error_t copy_from(const MemoryManager<T>& src);

    /** copies memory from a host ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    magmadnn_error_t copy_from_host(T* src, unsigned int begin_idx, unsigned int size);

#if defined(MAGMADNN_HAVE_CUDA)
    /** copies memory from a device ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    magmadnn_error_t copy_from_device(T* src, unsigned int begin_idx, unsigned int size);

    /** copies memory from a managed ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    magmadnn_error_t copy_from_managed(T* host_src, T* device_src, unsigned int begin_idx, unsigned int size);

    /** copies memory from a cuda managed ptr into this memorymanager. will throw an error if it
     *  reaches the end of src allocated mem before this is filled.
     *  @param src the array to copy into this.
     *  @return the error code (0 - good, 1 - not enough memory)
     */
    magmadnn_error_t copy_from_cudamanaged(T* src, unsigned int begin_idx, unsigned int size);
#endif

    /** If MANAGED or CUDA_MANAGED this ensures that data is the same on all devices. It
     * will wait for any gpu kernels to finish before copying data. If HOST or DEVICE memory
     * this does nothing.
     * @param gpu_was_modified If true then data will be copied from gpu to cpu, else if false vice-versa.
     * By default true.
     * @return an error code
     */
    magmadnn_error_t sync(bool gpu_was_modified = true);

    /** Changes the device this memory manager points to. Note that the memory type
     *  is still the same, but the device_id will be different.
     *  @return an error code (0 - ok)
     */
    magmadnn_error_t set_device(device_t device_id);

    /** Returns the value at idx. Error if idx is out of range.
     *  @param idx index to retrieve
     *  @return the value at index idx.
     */
    T get(unsigned int idx) const;

    /** Sets the value at idx to val. Error if idx is out of range.
     *  @param idx index to set
     *  @param val value to set at idx
     */
    void set(unsigned int idx, T val);

    /** returns a CPU pointer to the data.
     *  @return cpu pointer
     */
    T* get_host_ptr();

#if defined(MAGMADNN_HAVE_CUDA)
    /** returns a CUDA pointer
     *  @return a pointer to the memory on a cuda device.
     */
    T* get_device_ptr();

    /** returns the managed CUDA memory.
     *  @return pointer to data memory
     */
    T* get_cuda_managed_ptr();

    cudaStream_t get_custream() const { return this->custream_; }
#endif

    /** Returns a pointer to whatever memory type this is using. For MANAGED
     *  memory type it returns the device pointer.
     *  @return the data ptr
     */
    T* get_ptr();

#if defined(MAGMADNN_HAVE_CUDA)
    void set_custream(cudaStream_t custream) { this->custream_ = custream; }
#endif

    /** Returns the size of this memorymanager
     * @return unsigned int  the size of this memory manager
     */
    unsigned int get_size() const { return size; }

    /** Returns the memory type of this memory manager.
     * @return memory_t
     */
    memory_t get_memory_type() const { return mem_type; }

    /** Zero out memory
     *
     */
    magmadnn_error_t zero();

   private:
    /** init with HOST parameters */
    void init_host();

#if defined(MAGMADNN_HAVE_CUDA)
    /** init with DEVICE parameters */
    void init_device();

    /** init with MANAGED parameters */
    void init_managed();

    /** init with CUDA_MANAGED parameters */
    void init_cuda_managed();
#endif

    memory_t mem_type;
    device_t device_id;

    unsigned int size;
    T* host_ptr;

#if defined(MAGMADNN_HAVE_CUDA)
    T* device_ptr;
    T* cuda_managed_ptr;
    // CUDA stream used for GPU operations
    cudaStream_t custream_;
#endif
};

}  // namespace magmadnn
