#ifndef ROADSEGMENTS_H
#define ROADSEGMENTS_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class RoadSegments;
typedef std::shared_ptr<RoadSegments> RoadSegmentsPtr;

/**
 * Class for managing road segments
 */
class RoadSegments : public std::enable_shared_from_this<RoadSegments> {

public:
    // ENUMERATIONS ///////////////////////////////////////////////////////////
    typedef enum {
        ROAD,
        BRIDGE,
        TUNNEL
    } Type;
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor
     *
     * Constructs a %RoadSegments object with default values
     */
    RoadSegments(RoadPtr road);

    /**
     * Destructor
     */
    ~RoadSegments();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the road
     *
     * @return Road as RoadPtr
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
     * Returns the X coordinates
     *
     * @return X coordinates as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getX() {
        return this->x;
    }
    /**
     * Sets the X coordinates
     *
     * @param x as const Eigen::VectorXd&
     */
    void setX(const Eigen::VectorXd& x) {
        this->x = x;
    }

    /**
     * Returns the Y coordinates
     *
     * @return Y coordinates as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getY() {
        return this->y;
    }
    /**
     * Sets the Y coordinates
     *
     * @param y as const Eigen::VectorXd&
     */
    void setY(const Eigen::VectorXd& y) {
        this->y = y;
    }

    /**
     * Returns the natural elevation
     *
     * @return Natural elevation as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getE() {
        return this->e;
    }
    /**
     * Sets the natural elevation
     *
     * @param e as const Eigen::VectorXd&
     */
    void setE(const Eigen::VectorXd& e) {
        this->e = e;
    }

    /**
     * Returns the Z coordinates
     *
     * @return Z coordinates as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getZ() {
        return this->z;
    }
    /**
     * Sets the Z coordinates
     *
     * @param z as const Eigen::VectorXd&
     */
    void setZ(const Eigen::VectorXd& z) {
        this->z = z;
    }

    /**
     * Returns the parametrised distance along the curve
     *
     * @return Distances as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getDists() {
        return this->s;
    }
    /**
     * Sets the parametrised distance along the curve
     *
     * @param s as const Eigen::VectorXd&
     */
    void setDists(const Eigen::VectorXd& s) {
        this->s = s;
    }

    /**
     * Returns the widths
     *
     * @return Widths as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getWidths() {
        return this->w;
    }
    /**
     * Sets the segment widths
     *
     * @param w as const Eigen::VectorXd&
     */
    void setWidths(const Eigen::VectorXd& w) {
        this->w = w;
    }

    /**
     * Returns the velocities
     *
     * @return Velocities as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getVelocities() {
        return this->v;
    }
    /**
     * Sets the velocities
     *
     * @param v as const Eigen::VectorXd&
     */
    void setVelocities(const Eigen::VectorXd& v) {
        this->v = v;
    }

    /**
     * Returns the distances to curve points
     *
     * @return Distances as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getSPC() {
        return this->spc;
    }
    /**
     * Sets the distances to curve points
     *
     * @param spc as const Eigen::VectorXd&
     */
    void setSPC(const Eigen::VectorXd& spc) {
        this->spc = spc;
    }

    /**
     * Returns the type of road
     *
     * Although there is an enum for the three different types (ROAD,
     * BRIDGE and TUNNEL), we cast them as ints so that we may store
     * them in a matrix for easy use.
     *
     * @return Road type as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getType() {
        return this->typ;
    }
    /**
     * Sets the type of road
     *
     * @param typ as const Eigen::VectorXi&
     */
    void setType(const Eigen::VectorXi& typ) {
        this->typ = typ;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Computes the road segments for a road
     */
    void computeSegments();

    /**
     * Finds the natural elevation of a road at each design point. To use this,
     * the data must be in a regular rectangular grid. To achieve this, we must
     * first clean the input data provided earlier in the program.
     */
    void placeNetwork();

private:
    std::weak_ptr<Road> road;	/**< Road containing these segments */
    Eigen::VectorXd x;          /**< X coordinates */
    Eigen::VectorXd y;          /**< Y coordinates */
    Eigen::VectorXd z;          /**< Z coordinates */
    Eigen::VectorXd e;          /**< Natural elevation at x,y */
    Eigen::VectorXd s;          /**< Distance along curve */
    Eigen::VectorXd w;          /**< Road widths at each points */
    Eigen::VectorXd v;          /**< Velocity at point */
    Eigen::VectorXd spc;        /**< Distances to curve points */
    Eigen::VectorXi typ;        /**< Type of segments */

    // PRIVATE ROUTINES

    /**
     * Computes the overall road length
     */
    void computeRoadLength();
};

#endif
