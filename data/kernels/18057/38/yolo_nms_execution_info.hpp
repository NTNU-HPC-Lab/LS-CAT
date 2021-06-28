#ifndef __YOLO_NMS_EXECUTION_INFO_HPP__
#define __YOLO_NMS_EXECUTION_INFO_HPP__
#include "execution_info.hpp"


namespace tensorrtInference
{
    class YoloNMSExecutionInfo : public ExecutionInfo
    {
    public:
        YoloNMSExecutionInfo(CUDARuntime *runtime, 
            std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        ~YoloNMSExecutionInfo();
        bool init(Json::Value& root) override;
        void run() override;
        void callYoloNMSExecutionKernel();
        void recycleBuffers();
    private:
        std::shared_ptr<Buffer> sortIdxBuffer;
        std::shared_ptr<Buffer> sortProbBuffer;
        std::shared_ptr<Buffer> idxBuffer;
        std::shared_ptr<Buffer> probBuffer;
        std::shared_ptr<Buffer> maskBuffer;
        std::shared_ptr<Buffer> cubBuffer;
        std::shared_ptr<Buffer> maskRemoveBuffer;
        int boxesNum;
        int classesNum;
        int cubBufferSize;
        int imgHeight;
        int imgWidth;
        float confThresh;
        float iouThresh;
    };
} // namespace tensorrtInference 

#endif