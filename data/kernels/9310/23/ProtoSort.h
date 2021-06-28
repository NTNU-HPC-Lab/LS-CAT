#pragma once


namespace proto
{


template<typename TData>
void selectionSortCpu(const TData* data, unsigned int count);

//template<typename TData>
//void bubbleSortCpu(const TData* data, unsigned int count)
//{
//    for (int i = count - 1; i > 0; i--)
//    {
//        for (int j = i - 1; j >= 0; )
//    }
//}
//
//template<typename TData>
//void bubbleSortGpu(const TData* data, unsigned int count);
//
//template<typename TData>
//void mergeSortCpu(const TData* data, unsigned int count);
//
//template<typename TData>
//void mergeSortGpu(const TData* data, unsigned int count);
//
//template<typename TData>
//void quickSortCpu(const TData* data, unsigned int count);
//
//template<typename TData>
//void quickSortGpu(const TData* data, unsigned int count);


}
