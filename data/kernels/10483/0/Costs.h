#ifndef COSTS_H
#define COSTS_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class RoadCells;
typedef std::shared_ptr<RoadCells> RoadCellsPtr;

class RoadSegments;
typedef std::shared_ptr<RoadSegments> RoadSegmentsPtr;

class Species;
typedef std::shared_ptr<Species> SpeciesPtr;

class SpeciesRoadPatches;
typedef std::shared_ptr<SpeciesRoadPatches> SpeciesRoadPatchesPtr;

class HabitatType;
typedef std::shared_ptr<HabitatType> HabitatTypePtr;

class UnitCosts;
typedef std::shared_ptr<UnitCosts> UnitCostsPtr;

class Vehicle;
typedef std::shared_ptr<Vehicle> VehiclePtr;

class Costs;
typedef std::shared_ptr<Costs> CostsPtr;

/**
 * Class for managing road costs
 */
class Costs : public std::enable_shared_from_this<Costs> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a default cost object
     */
    Costs(RoadPtr road);

    /**
     * Constructor II
     *
     * Constructs a cost object with assigned values
     */
    Costs(RoadPtr road, double af, double av, double e, double lf, double lv,
            double loc, double pc);

    /**
     * Destructor
     */
    ~Costs();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the road
     *
     * @return Road fixed costs as RoadPtr
     */
    RoadPtr getRoad() {
        return this->road.lock();
    }
    /**
     * Sets the road
     *
     * @param road as RoadPtr
     */
    void setRoad(RoadPtr road) {
    this->road.reset();
        this->road = road;
    }

    /**
     * Returns the fixed component of accident costs
     *
     * @return Accident fixed costs as double
     */
    double getAccidentFixed() {
        return this->accidentFixed;
    }
    /**
     * Sets the fixed component of accident costs
     *
     * @param af as double
     */
    void setAccidentFixed(double af) {
        this->accidentFixed = af;
    }

    /**
     * Returns the variable component of accident costs
     *
     * @return Accident variable costs as double
     */
    double getAccidentVariable() {
        return this->accidentVar;
    }
    /**
     * Sets the variable component of accident costs
     *
     * @param av as double
     */
    void setAccidentVariable(double av) {
        this->accidentVar = av;
    }

    /**
     * Returns earthwork costs
     *
     * @return Earthwork costs as double
     */
    double getEarthwork() {
        return this->earthwork;
    }
    /**
     * Sets earthwork costs
     *
     * @param e as double
     */
    void setEarthwork(double e) {
        this->earthwork = e;
    }

    /**
     * Returns fixed length-based costs
     *
     * @return Fixed length-based as double
     */
    double getLengthFixed() {
        return this->lengthFixed;
    }
    /**
     * Sets fixed length-based costs
     *
     * @param lf as double
     */
    void setLengthFixed(double lf) {
        this->lengthFixed = lf;
    }

    /**
     * Returns variable length-based costs
     *
     * @return Variable length-based costs as double
     */
    double getLengthVariable() {
        return this->lengthVar;
    }
    /**
     * Sets variable length-based costs
     *
     * @param lv as double
     */
    void setLengthVariable(double lv) {
        this->lengthVar = lv;
    }

    /**
     * Returns location-based costs
     *
     * @return Location-based costs as double
     */
    double getLocation() {
        return this->location;
    }
    /**
     * Sets location-based costs
     *
     * @param loc as double
     */
    void setLocation(double loc) {
        this->location = loc;
    }

    /**
     * Returns animal mortality penalty costs
     *
     * @return Animal mortality penalty costs as double
     */
    double getPenalty() {
        return this->penaltyCost;
    }
    /**
     * Sets animal mortality penalty costs
     *
     * @param penalty as double
     */
    void setPenalty(double penalty) {
        this->penaltyCost = penalty;
    }

    /**
     * Returns the annualised fuel usage per unit traffic per hour per vehicle class
     *
     * @return Unit fuel cost as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getUnitFuelCost() {
        return this->unitFuelVar;
    }
    /**
     * Returns the annualised fuel usage per unit traffic per hour per vehicle class
     *
     * @param fuel as const Eigen::VectorXd&
     */
    void setUnitFuelCost(const Eigen::VectorXd& fuel) {
        this->unitFuelVar = fuel;
    }

    /**
     * Returns the revenue per unit traffic
     *
     * @return Unit revenue as double
     */
    double getUnitRevenue() {
        return Costs::unitRevenueVar;
    }
    /**
     * Sets the revenue per unit traffic
     *
     * @param rev as double
     */
    void setUnitRevenue(double rev) {
        Costs::unitRevenueVar = rev;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    /**
     * Computes the carrying amount of ore per unit traffic
     */
    static void computeUnitRevenue(OptimiserPtr optimiser);

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Computes the earthwork costs for building the road as well as the type
     * of each segment:
     * 1 = regular road section
     * 2 = bridge section
     * 3 = tunnel section
     */
    void computeEarthworkCosts();

    /**
     * Computes the location-based costs
     *
     * For the purposes of computing these costs etc, we treat the road segment
     * between each of the input x,y coordinate pairs as straight lines. This
     * differs from the actual computation of road lengths in other functions
     * but we do not require the same accuracy here as the costs themselves are
     * somewhat inaccurate.
     *
     * Lengths is a vector of the length of the road in various habitat types:
     * 1 - non-habitat
     * 2 - marginal
     * 3 - secondary
     * 4 - primary
     */
    void computeLocationCosts();

    /**
     * Computes fixed and variable length-based costs
     */
    void computeLengthCosts();

    /**
     * Computes fixed and variable accident-based costs
     */
    void computeAccidentCosts();

    /**
     * Computes the cost of the end population being below threshold
     *
     * @note This only applies to the design case where the road is run at full
     * capacity for the entire design horizon
     */
    void computePenaltyCost();

private:
    std::weak_ptr<Road> road;           /**< Road with these costs */
    double accidentFixed;		/**< Fixed component of accident cost */
    double accidentVar;			/**< Variable accident cost per unit traffic */
    double earthwork;			/**< Total earthworks cost */
    double lengthFixed;			/**< Fixed costs related to length */
    double lengthVar;			/**< Variable length costs per unit traffic */
    double location;			/**< Location-based fixed costs */
    double penaltyCost;			/**< Animal mortality penalty cost */
    Eigen::VectorXd unitFuelVar;        /**< Fuel usage per unit traffic */
    static double unitRevenueVar;       /**< Haul load per unit traffic */
};

#endif
