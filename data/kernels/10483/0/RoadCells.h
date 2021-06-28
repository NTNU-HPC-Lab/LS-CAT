#ifndef ROADCELLS_H
#define ROADCELLS_H

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class RoadCells;
typedef std::shared_ptr<RoadCells> RoadCellsPtr;

/**
 * Class for managing the grid cells occupied by a road
 */
class RoadCells : public std::enable_shared_from_this<RoadCells> {

public:

    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructs and object with the details of the cells occupied by a road.
     */
    RoadCells(RoadPtr road);

    /**
     * Destructor
     */
    ~RoadCells();

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
     * Returns the x values
     *
     * @return X values as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getX() {
    return this->x;
    }
    /**
     * Sets the x values of the segments
     *
     * @param x as const Eigen::VectorXd&
     */
    void setX(const Eigen::VectorXd& x) {
        this->x = x;
    }

    /**
     * Returns the y values
     *
     * @return Y values as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getY() {
        return this->y;
    }
    /**
     * Sets the y values of the segments
     *
     * @param y as const Eigen::VectorXd&
     */
    void setY(const Eigen::VectorXd& y) {
        this->y = y;
    }

    /**
     * Returns the z values
     *
     * @return Z values as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getZ() {
        return this->z;
    }
    /**
     * Sets the z values of the segments
     *
     * @param z as const Eigen::VectorXd&
     */
    void setZ(const Eigen::VectorXd& z) {
        this->z = z;
    }

    /**
     * Returns the width values
     *
     * @return Widths as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getWidths() {
        return this->w;
    }
    /**
     * Sets the width values of the segments
     *
     * @param w as const Eigen::VectorXd&
     */
    void setWidths(const Eigen::VectorXd& w) {
        this->w = w;
    }

    /**
     * Returns the vegetation at each point
     *
     * @return Vegetation as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getVegetation() {
        return this->veg;
    }
    /**
     * Sets the vegetation at each point
     *
     * @param veg as const Eigen::VectorXi&
     */
    void setVegetation(const Eigen::VectorXi& veg) {
        this->veg = veg;
    }

    /**
     * Returns the segement lengths
     *
     * @return Segment lengths as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getLengths() {
        return this->len;
    }
    /**
     * Sets the segment lengths
     *
     * @param len as const Eigen::VectorXd&
     */
    void setLengths(const Eigen::VectorXd& len) {
        this->len = len;
    }

    /**
     * Returns the road segment areas
     *
     * @return Areas as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getAreas() {
        return this->areas;
    }
    /**
     * Sets the road semgnet areas
     *
     * @param areas as const Eigen::VectorXd&
     */
    void setAreas(const Eigen::VectorXd& areas) {
        this->areas = areas;
    }

    /**
     * Returns the road semgnet types
     *
     * @return Tyeps as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getTypes() {
        return this->type;
    }
    /**
     * Sets the types of the segments
     *
     * @param type as const Eigen::VectorXi&
     */
    void setTypes(const Eigen::VectorXi& type) {
        this->type = type;
    }

    /**
     * Returns the cell references
     *
     * Cells are referenced in (xcoord, ycoord), starting at coordinate (0,0)
     *
     * @return Cell coordinates as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getCellRefs() {
        return this->cellRefs;
    }
    /**
     * Sets the cell references
     *
     * Cells are referenced as indices in column-major format
     *
     * @param cellrefs as const Eigen::VectorXi&
     */
    void setCellRefs(const Eigen::VectorXi& cellrefs) {
        this->cellRefs = cellrefs;
    }

    /**
     * Returns the unique cells occupied by the road
     *
     * @return Cells as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getUniqueCells() {
        return this->uniqueCells;
    }
    /**
     * Sets the unique cells occupied by the road
     *
     * @param cells as const Eigen::VectorXi&
     */
    void setUniqueCells(const Eigen::VectorXi& cells) {
        this->uniqueCells = cells;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Computes the grid cells to which the road belongs
     *
     * The RoadCells object contains
     *
     * X positions
     * Y positions
     * Z positions
     * Length of segments
     * Area of segments
     * Type (road, bridge, tunnel) of segments
     * Cell references (x,y positions)
     * List of unique cells
     */
    void computeRoadCells();

private:
    std::weak_ptr<Road> road;       /**< Road */
    Eigen::VectorXd x;              /**< X values */
    Eigen::VectorXd y;              /**< Y values */
    Eigen::VectorXd z;              /**< Z values */
    Eigen::VectorXd w;              /**< Segment widths */
    Eigen::VectorXi veg;            /**< Vegetation type */
    Eigen::VectorXd len;            /**< Length of each section */
    Eigen::VectorXd areas;          /**< Area of each section */
    Eigen::VectorXi type;           /**< Type of road section */
    Eigen::VectorXi cellRefs;       /**< Corresponding cell references (column-major indices) */
    Eigen::VectorXi uniqueCells;    /**< Same as above, duplicates removed */
};

#endif

