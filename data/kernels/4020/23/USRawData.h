// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __USRAWDATA_H__
#define __USRAWDATA_H__

#include <memory>
#include <stddef.h>

#include "Container.h"
#include "RecordObject.h"
#include "USImageProperties.h"
#include "RxBeamformerParameters.h"

namespace supra
{
	class USRawData : public RecordObject
	{
	public:
		USRawData(size_t numScanlines,
			size_t numElements,
			vec2s elementLayout,
			size_t numReceivedChannels,
			size_t numSamples,
			double samplingFrequency,
			std::shared_ptr<const ContainerBase > pData,
			std::shared_ptr<const RxBeamformerParameters> pRxBeamformerParameters,
			std::shared_ptr<const USImageProperties> pImageProperties,
			double receiveTimestamp,
			double syncTimestamp)
			: RecordObject(receiveTimestamp, syncTimestamp)
			, m_numScanlines(numScanlines)
			, m_numElements(numElements)
			, m_elementLayout(elementLayout)
			, m_numReceivedChannels(numReceivedChannels)
			, m_numSamples(numSamples)
			, m_samplingFrequency(samplingFrequency)
			, m_pData(pData)
			, m_pRxBeamformerParameters(pRxBeamformerParameters)
			, m_pImageProperties(pImageProperties)
		{};

		std::shared_ptr<const USImageProperties> getImageProperties() const { return m_pImageProperties; };
		/// Sets the \see USImageProperties that contain the associated metadata
		void setImageProperties(std::shared_ptr<USImageProperties> & imageProperties) { m_pImageProperties = imageProperties; };
		template <typename ElementType>
		std::shared_ptr<const Container<ElementType> > getData() const 
		{
			return std::dynamic_pointer_cast<const Container<ElementType> >(m_pData); 
		}
		size_t getNumScanlines() const { return m_numScanlines; };
		size_t getNumElements() const { return m_numElements; };
		vec2s getElementLayout() const { return m_elementLayout; };
		size_t getNumReceivedChannels() const { return m_numReceivedChannels; };
		size_t getNumSamples() const { return m_numSamples; };
		double getSamplingFrequency() const { return m_samplingFrequency; };
		std::shared_ptr<const RxBeamformerParameters> getRxBeamformerParameters() const { return m_pRxBeamformerParameters; };

		virtual RecordObjectType getType() const { return TypeUSRawData; }
		DataType getDataType() const { return m_pData->getType(); }

	private:
		size_t m_numScanlines;
		size_t m_numElements;
		vec2s m_elementLayout;
		size_t m_numReceivedChannels;
		size_t m_numSamples;
		double m_samplingFrequency; // [MHz]
		std::shared_ptr<const ContainerBase> m_pData;
		std::shared_ptr<const RxBeamformerParameters> m_pRxBeamformerParameters;

		std::shared_ptr<const USImageProperties> m_pImageProperties;
	};
}

#endif //!__USRAWDATA_H__
