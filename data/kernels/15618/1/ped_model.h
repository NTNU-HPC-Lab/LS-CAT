//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <ppl.h>
#include <atomic>
#include <chrono>

#include "ped_agent.h"
#include "ped_region.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation, IMPLEMENTATION moveImp);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// compute and update next positions for the next 4 agents starting from "index"
		void computeNextPosition(int start, int end);

		// Returns the agents of this scenario
		const std::vector<Tagent*> getAgents() const { return agents; };

		// Return the agents' X coordinate vector reference
		const std::vector<float>& getAgentsX() const { return agentsX; }

		// Return the agents' Y coordinate vector reference
		const std::vector<float>& getAgentsY() const { return agentsY; }

		void setPositionNoCollision(std::vector<std::pair<float, float> > prioritizedAlternatives, int i, int j);

		std::vector<std::pair<float, float>> computeAlternative(int i);

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		int const * cudaGetHeatmap() const { return cuda_blurred_heatmap; };
	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		IMPLEMENTATION moveImp;

		// The agents in this scenario
		std::vector<Tagent*> agents;
		
		// store agents coordinates into 2 vectors
		std::vector<float> agentsX;

		std::vector<float> agentsY;

		// desired position array
		std::vector<float> desiredAgentsX;

		std::vector<float> desiredAgentsY;

		// store the waypoints for each agent into a 2D vector
		std::vector<std::deque<Twaypoint*>> waypoints;

		// store the current destination coordinate and the radius into 3 vectors
		std::vector<float> destX;

		std::vector<float> destY;

		std::vector<float> destR;

		int* reached;

		std::vector<std::vector<int>> regionAgentList;

		std::vector<Ped::Region> regionList;

		std::vector<std::atomic<double>> agentsIsBeingProcessed;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void moveSeq();
		void move();

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		/** 
		*	SEQ heatmap variables
		*/
		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();

		/**
		*  CUDA heatmap variable
		*/
		int* cuda_blurred_heatmap;
	};
}
#endif
