#pragma once
#include "IResource.h"
#include "ResourceManager.h"
#include "NodesDataHaloed.h"

namespace TrailEvolutionModelling {
	namespace GPUProxy {

		template <typename T> struct NodesDataHaloedDevice;

		template<typename T>
		struct NodesDataDevicePair : public IResource {
			friend class ResourceManager;

			NodesDataHaloedDevice<T>* readOnly = nullptr;
			NodesDataHaloedDevice<T>* writeOnly = nullptr;

#ifndef __CUDACC__
			void CopyReadToWriteSync(int graphW, int graphH);
			void CopyWriteToReadSync(int graphW, int graphH);
			void CopyReadToWrite(int graphW, int graphH, cudaStream_t stream);
			void CopyWriteToRead(int graphW, int graphH, cudaStream_t stream);

		protected:
			NodesDataDevicePair(int graphW, int graphH, ResourceManager* resources);
			void Free(ResourceManager& resources) override;
#endif
		};



#ifndef __CUDACC__
		template<typename T>
		inline NodesDataDevicePair<T>::NodesDataDevicePair(int graphW, int graphH, ResourceManager* resources) 
			: readOnly(resources->New<NodesDataHaloedDevice<T>>(graphW, graphH)),
			  writeOnly(resources->New<NodesDataHaloedDevice<T>>(graphW, graphH)) 
		{
		}


		template<typename T>
		inline void NodesDataDevicePair<T>::CopyReadToWriteSync(int graphW, int graphH) {
			size_t size = NodesDataHaloed<T>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemcpy(writeOnly->data, readOnly->data, size, cudaMemcpyDeviceToDevice));
		}

		template<typename T>
		inline void NodesDataDevicePair<T>::CopyWriteToReadSync(int graphW, int graphH) {
			size_t size = NodesDataHaloed<T>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemcpy(readOnly->data, writeOnly->data, size, cudaMemcpyDeviceToDevice));
		}

		template<typename T>
		inline void NodesDataDevicePair<T>::CopyReadToWrite(int graphW, int graphH, cudaStream_t stream) {
			size_t size = NodesDataHaloed<T>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemcpyAsync(writeOnly->data, readOnly->data, size, cudaMemcpyDeviceToDevice, stream));
		}

		template<typename T>
		inline void NodesDataDevicePair<T>::CopyWriteToRead(int graphW, int graphH, cudaStream_t stream) {
			size_t size = NodesDataHaloed<T>::ArraySizeBytes(graphW, graphH);
			CHECK_CUDA(cudaMemcpyAsync(readOnly->data, writeOnly->data, size, cudaMemcpyDeviceToDevice, stream));
		}

		template<typename T>
		inline void NodesDataDevicePair<T>::Free(ResourceManager& resources) {
			resources.Free(readOnly);
			resources.Free(readOnly);
		}
#endif

	}
}