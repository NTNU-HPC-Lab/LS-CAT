#ifndef __EXCUTION_INFO_HPP__
#define __EXCUTION_INFO_HPP__
#include <vector>
#include "cuda_runtime.hpp"
#include "buffer.hpp"
#include "utils.hpp"
#include "json/json.h"
#define DIVUP(m,n) (((m)+(n)-1) / (n))

namespace tensorrtInference {

    class ExecutionInfo {
    public:
        ExecutionInfo(CUDARuntime *runtime, std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        virtual ~ExecutionInfo();
        virtual bool init(Json::Value& root) = 0;
        virtual void run() = 0;
        void printExecutionInfo();
        const std::string getExecutionInfoType() {return executionInfoType;}
        const std::vector<std::string>& getInputTensorNames() {return inputTensorNames;}
        const std::vector<std::string>& getOutputTensorNames() {return outputTensorNames;}
        CUDARuntime* getCudaRuntime() {return cudaRuntime;}
        void initTensorInfo(std::map<std::string, std::shared_ptr<Buffer>>& tensorsInfo, Json::Value& root);
        const std::map<std::string, Buffer*>& getTensorsInfo() { return tensors; }
        Buffer* mallocBuffer(int size, OnnxDataType dataType, bool mallocHost, bool mallocDevice, 
            StorageType type = StorageType::DYNAMIC);
        Buffer* mallocBuffer(std::vector<int> shape, OnnxDataType dataType, bool mallocHost, 
            bool mallocDevice, StorageType type = StorageType::DYNAMIC);
        void recycleBuffers();
        void beforeRun();
        void afterRun();
        template<typename T>
        void printBuffer(Buffer* buffer, int start, int end)
        {
            
            CHECK_ASSERT(buffer != nullptr, "buffer must not be none!\n");
            CHECK_ASSERT(start >= 0, "start index must greater than 1!\n");
            auto shape = buffer->getShape();
            auto dataType = buffer->getDataType();
            std::shared_ptr<Buffer> debugBuffer(mallocBuffer(shape, dataType, true, false));
            cudaRuntime->copyFromDevice(buffer, debugBuffer.get());
            cudaError_t cudastatus = cudaGetLastError();
            CHECK_ASSERT(cudastatus == cudaSuccess, "launch memcpy kernel fail: %s\n", cudaGetErrorString(cudastatus));
            
            auto debugData = debugBuffer->host<T>();
            int count = debugBuffer->getElementCount();
            int printStart = (start > count) ? 0 : start;
            int printEnd   = ((end - start) > count) ? (start + count) : end;
            std::cout << "buffer data is :" << std::endl;
            if(onnxDataTypeEleCount[buffer->getDataType()] != 1)
            {
                for(int i = printStart; i < printEnd; i++)
                {
                    std::cout << debugData[i] << " " << std::endl;
                }
            }
            else
            {
                for(int i = printStart; i < printEnd; i++)
                {
                    printf("%d \n", debugData[i]);
                }
            }
        }        
    private:
        CUDARuntime* cudaRuntime;
        std::string executionInfoType;
        std::map<std::string, Buffer*> tensors;
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
        std::map<std::string, std::string> memcpyDir;
    };

    typedef ExecutionInfo* (*constructExecutionInfoFunc)(CUDARuntime *runtime, 
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);

    class ConstructExecutionInfo
    {
    private:
        static ConstructExecutionInfo* instance;
        void registerConstructExecutionInfoFunc();
        std::map<std::string, constructExecutionInfoFunc> constructExecutionInfoFuncMap;
        ConstructExecutionInfo()
        {
        }
    public:
        constructExecutionInfoFunc getConstructExecutionInfoFunc(std::string executionType);
        static ConstructExecutionInfo* getInstance() {
            return instance;
        }
    };
    
    extern constructExecutionInfoFunc getConstructExecutionInfoFuncMap(std::string executionType);

}
#endif