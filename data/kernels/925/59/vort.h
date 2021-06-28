//##############################################################################
/**
 *  @file    vort.h
 *  @author  Lee J. O'Riordan (mlxd)
 *  @date    12/11/2015
 *  @version 0.1
 *
 *  @brief Class for keeping track of vortices. Implementation TBC
 *
 *  @section DESCRIPTION
 *  Each vortex is treated as a struct of position, winding, and least-squares
 *  calculated positions. UID, intervortex separations, and lifetime can be
 *  tracked and maintained.
 */
 //#############################################################################

#ifndef GPUE_1_VORT_H
#define GPUE_1_VORT_H

#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <cmath>

#include<iostream>
#include <algorithm>
#include <set>
#include <utility>      // std::pair, std::make_pair

//@todo Convert the GPUE codebase to use the newly implemented Vortex and VtxList classes. Simplify graph codebase
namespace Vtx {

	/**
	* Maintains vortex index in grid and least-squares calculated values
	* with winding/direction of vortex rotation.
	*/
    class Vortex {
        friend class VtxList;
    private:
	    int2 coords;
	    double2 coordsD;
	    int winding;
	    bool isOn;
	    std::size_t timeStep;

    protected:
        int uid;

    public:
        Vortex();
        Vortex(int2 coords, double2 coordsD, int winding, bool isOn, std::size_t timeStep);

        ~Vortex();

        void updateUID(int uid);
        void updateWinding(int winding);
        void updateIsOn(bool isOn);
        void updateCoords(int2 coords);
        void updateCoordsD(double2 coordsD);
        void updateTimeStep(std::size_t timeStep);

        int getUID() const;
        int getWinding() const;
	    bool getIsOn() const;
        int2 getCoords() const;
        double2 getCoordsD() const;
        std::size_t getTimeStep() const;
    };


// Adding block comments in the least intuitive way because of doxygen comments
//##############################################################################
	/**
	* Vortex list for storing and retrieving vortices and associated values
	*/
	class VtxList {
		private:
			std::vector< std::shared_ptr<Vtx::Vortex> > vortices;
            std::size_t suid; //value of max UID. Used to allow for independent UIDs upon creation.
		public:
            VtxList(); //Initialise suid when creating the list
            VtxList(std::size_t reserveSize);//If size is known in advance, reserve the required num of elements
            ~VtxList(); //Remove all shared_ptr values

            /**
            * @brief	Adds a vortex to the list
            * @ingroup	vtx
            * @return	Reference to vortex list
            */
            void addVtx(std::shared_ptr<Vtx::Vortex> vtx);
            void addVtx(std::shared_ptr<Vtx::Vortex> vtx, std::size_t idx);

            /**
            * @brief	Remove a vortex at position idx
            * @ingroup	vtx
            * @return	The shared_ptr pointer to the removed vortex
            */
            std::shared_ptr<Vortex> removeVtx(std::size_t idx);

            /**
            * @brief	Returns a reference to the vortex list
            * @ingroup	vtx
            * @return	Reference to vortex list
            */
            std::vector< std::shared_ptr<Vortex> > &getVortices();

			/**
			* @brief	Returns a shared_ptr to the vortex by a UID
			* @ingroup	vtx
			* @return	shared_ptr<Vortex> Shared pointer to vortex by index
			*/
			std::shared_ptr<Vortex> getVtx_Uid(int uid);

			/**
			* @brief	Returns a shared_ptr to the vortex by an index in VtxList
			* @ingroup	vtx
			* @return	shared_ptr<Vortex> Shared pointer to vortex by index
			*/
			std::shared_ptr<Vortex> getVtx_Idx(std::size_t idx);

			/**
			* @brief	Returns a vortex index based upon a given vortex UID
			* @ingroup	vtx
			* @return	unsigned int Vortex index for UID
			*/
            std::size_t  getVtxIdx_Uid(int uid);

			/**
			* @brief	Returns the largest UID given
			* @ingroup	vtx
			* @return	unsigned int Largest vortex UID
			*/
            std::size_t&  getMax_Uid();

			/**
			* @brief	Returns index of vortex with shortest coordinate distance from current vortex
			* @ingroup	vtx
			* @return	unsigned int Vortex index for shortest distance
			*/
            std::shared_ptr<Vortex> getVtxMinDist(std::shared_ptr<Vortex> vtx);

			/**
			* @brief	In-place swap of the UID for the two given vortices
			* @ingroup	vtx
			* @param	v1 Shared pointer of Vortex v1
			* @param	v2 Shared pointer of Vortex v2
			*/
            void swapUid(std::shared_ptr<Vortex> v1, std::shared_ptr<Vortex> v2);
            /**
            * @brief	In-place swap of the UID for the two given vortices
            * @ingroup	vtx
            * @param	idx0 Index pointer of Vortex v0
            * @param	idx1 Index pointer of Vortex v1
            */
            void swapUid_Idx(std::size_t idx0, std::size_t idx1);

			/**
			* @brief	Turns vortex activation off. Useful if vortex no longer exists in condensate.
			* @ingroup	vtx
			*/
			void vortOff();

            /**
            * @brief	Sorts the vortices based on UID, from low to high
            * @ingroup	vtx
            * @param    cArray Vortex
            */
            void sortVtxUID();

            /**
            * @brief	Arrange the vortices to have correct UID corresponding across timesteps
            * @ingroup	vtx
            * @param    cArray Vortex
            */
            void arrangeVtx(std::vector<std::shared_ptr<Vortex> > &vPrev);


            void setUIDs(std::set<std::shared_ptr<Vtx::Vortex> > &v);


            std::pair<double,std::shared_ptr<Vortex> > minDistPair(std::shared_ptr<Vortex> vtx, double minRange);


        };
};
//##############################################################################
#endif //GPUE_1_VORT_H
