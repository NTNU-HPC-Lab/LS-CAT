#pragma once

/**

CUDA Device Memory RAII principle classes

New unified memory classes are used where applicable
These are wrapper of cuda functions and their most important responsiblity is
to delete allocated memory

All of the operations (execpt allocation) are asyncronious.

TODO: should we interface these?

*/
#include <limits>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

// Basic semi-interface for memories that are static for each GPU
// Textures are one example
class DeviceLocalMemoryI
{
    protected:
        int                     currentDevice;

    public:
                                DeviceLocalMemoryI(int initalDevice = 0) : currentDevice(initalDevice) {}
        virtual                 ~DeviceLocalMemoryI() = default;

        // Interface
        virtual void            MigrateToOtherDevice(int deviceTo, cudaStream_t stream = (cudaStream_t)0) = 0;
};

// Has a CPU Image of current memory
// Usefull for device static memory that can be generated at CPU while
// GPU doing work on GPU memory
// in our case some form of function backed animation can be calculated using these)
class DeviceMemoryCPUBacked : public DeviceLocalMemoryI
{
    private:
        void*                       h_ptr;
        void*                       d_ptr;

        size_t                      size;

    protected:
    public:
        // Constructors & Destructor
                                    DeviceMemoryCPUBacked();
                                    DeviceMemoryCPUBacked(size_t sizeInBytes, int deviceId = 0);
                                    DeviceMemoryCPUBacked(const DeviceMemoryCPUBacked&);
                                    DeviceMemoryCPUBacked(DeviceMemoryCPUBacked&&);
                                    ~DeviceMemoryCPUBacked();
        DeviceMemoryCPUBacked&      operator=(const DeviceMemoryCPUBacked&);
        DeviceMemoryCPUBacked&      operator=(DeviceMemoryCPUBacked&&);

        // Memcopy
        void                        CopyToDevice(size_t offset = 0, size_t copySize = std::numeric_limits<size_t>::max(), cudaStream_t stream = (cudaStream_t)0);
        void                        CopyToHost(size_t offset = 0, size_t copySize = std::numeric_limits<size_t>::max(), cudaStream_t stream = (cudaStream_t)0);

        // Access
        template<class T>
        constexpr T*                DeviceData();
        template<class T>
        constexpr const T*          DeviceData() const;
        template<class T>
        constexpr T*                HostData();
        template<class T>
        constexpr const T*          HostData() const;
        // Misc
        size_t                      Size() const;
        // Interface
        void                        MigrateToOtherDevice(int deviceTo, cudaStream_t stream = (cudaStream_t)0);
};

// Generic Device Memory (most of the cases this should be used)
// Fire and forget type memory
// In our case rays and hit records will be stored in this form
class DeviceMemory
{
    private:
        void*                       m_ptr;  // managed pointer

        size_t                      size;

    protected:
    public:
        // Constructors & Destructor
                                    DeviceMemory();
                                    DeviceMemory(size_t sizeInBytes);
                                    DeviceMemory(const DeviceMemory&);
                                    DeviceMemory(DeviceMemory&&);
                                    ~DeviceMemory();
        DeviceMemory&               operator=(const DeviceMemory&);
        DeviceMemory&               operator=(DeviceMemory&&);

        // Access
        template<class T>
        constexpr explicit          operator T*();
        template<class T>
        constexpr explicit          operator const T*() const;
        constexpr                   operator void*();
        constexpr                   operator const void*() const;
        // Misc
        size_t                      Size() const;
};

template<class T>
inline constexpr T* DeviceMemoryCPUBacked::DeviceData()
{
    return reinterpret_cast<T*>(d_ptr);
}

template<class T>
inline constexpr const T* DeviceMemoryCPUBacked::DeviceData() const
{
    return reinterpret_cast<T*>(d_ptr);
}

template<class T>
inline constexpr T* DeviceMemoryCPUBacked::HostData()
{
    return reinterpret_cast<T*>(h_ptr);
}

template<class T>
inline constexpr const T* DeviceMemoryCPUBacked::HostData() const
{
    return reinterpret_cast<T*>(h_ptr);
}

template<class T>
inline constexpr DeviceMemory::operator T*()
{
    return reinterpret_cast<T*>(m_ptr);
}

template<class T>
inline constexpr DeviceMemory::operator const T*() const
{
    return reinterpret_cast<T*>(m_ptr);
}

inline constexpr DeviceMemory::operator void*()
{
    return m_ptr;
}

inline constexpr DeviceMemory::operator const void*() const
{
    return m_ptr;
}