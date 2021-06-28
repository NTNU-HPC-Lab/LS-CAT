#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include "assert.h"


#define CHECK_ASSERT(x, format, args...) do {   \
    if(!(x)) {                                  \
        printf(format, ##args);                 \
        assert(0);                              \
    }                                           \
} while(0)

#define LOG(format, args...) do {               \
        printf(format, ##args);                 \
} while(0)


#define CUDA_CHECK(_x)                                       \
    do {                                                     \
        cudaError_t _err = (_x);                             \
        if (_err != cudaSuccess) {                           \
            CHECK_ASSERT(_err, #_x);                         \
        }                                                    \
    } while (0)

#define CUBLAS_CHECK(_x)                                     \
    do {                                                     \
        cublasStatus_t _err = (_x);                          \
        if (_err != CUBLAS_STATUS_SUCCESS) {                 \
            CHECK_ASSERT(_err, #_x);                         \
        }                                                    \
    } while (0)

#define CUSOLVER_CHECK(_x)                                   \
    do {                                                     \
        cusolverStatus_t _err = (_x);                        \
        if (_err != CUSOLVER_STATUS_SUCCESS) {               \
            CHECK_ASSERT(_err, #_x);                         \
        }                                                    \
    } while (0)

#define AFTER_KERNEL_LAUNCH()                                \
    do {                                                     \
        CUDA_CHECK(cudaGetLastError());                      \
    } while (0)

namespace tensorrtInference
{

    class Logger : public nvinfer1::ILogger
    {
    public:

        Logger(): Logger(Severity::kWARNING) {}

        Logger(Severity severity): reportableSeverity(severity) {}

        void log(Severity severity, const char* msg) override
        {
            // suppress messages with severity enum value greater than the reportable
            if (severity > reportableSeverity) return;

            switch (severity)
            {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }

        Severity reportableSeverity{Severity::kWARNING};
    };

    enum OnnxDataType {
        UNDEFINED,
        FLOAT,
        UINT8,
        INT8,
        UINT16,
        INT16,
        INT32,
        INT64,
        STRING,
        BOOL,
        FLOAT16,
        DOUBLE,
        UINT32,
        UINT64,
        COMPLEX64,
        COMPLEX128,
        BFLOAT16,
    };

    extern int onnxDataTypeEleCount[];
    extern std::vector<float> parseFloatArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape);
    extern std::vector<int> parseIntArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape);
    extern int getTensorrtDataType(OnnxDataType onnxDataType);
    extern std::vector<int> dimsToVector(nvinfer1::Dims dims);
    extern nvinfer1::Dims vectorToDims(std::vector<int> shape);
} //tensorrtInference
#endif