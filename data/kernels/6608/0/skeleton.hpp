#ifndef SKELETON_HPP__
#define SKELETON_HPP__

#include <vector>
#include <set>
#include <map>

#include "bone.hpp"
#include "joint_type.hpp"
#include "transfo.hpp"
#include "blending_lib/controller.hpp"
#include "skeleton_env_type.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// -----------------------------------------------------------------------------

/**
 *  @class Skeleton
    @brief The class that defines an articulated skeleton.
    Skeleton is represented by a tree of bones with a single joint as root
    and others as node and leaves

    We refer to the rest position or bind pose of the skeleton
    to define the initial position (i.e the moment when the skeleton/graph is
    attached to the mesh). Transformed position of the skeleton and mesh is
    the animated position computed according to the rest/bind position.

    Each joint defines the next bone except for the leaves which are bones with
    a length equal to zero :
    @code

    joint_0  bone_0   joint_1   bone_1   joint_3
    +------------------+------------------+
                     joint_2
                        \
                         \ bone_2
                          \
                           + joint_4

    @endcode
    In this example we can see that joint_1 and joint_2 has the same position
    but not the same orientation. joint_3 and joint_4 are the leaves of the
    skeleton tree the corresponding  bone length cannot be defined.
    Note that a joint frame (fetched with 'joint_anim_frame()' for instance) is
    usually different from the bone frame ( bone_anim_frame() )

    One way to look up the skeleton's bones is to explore the bone array :
    @code
    Skeleton skel;
    ...
    for(int i = 0; i < skel.get_nb_joints(); i++){

        if( skel.is_leaf(i) )
        {
            // Do something or skip it
        }

        // Acces the ith bone :
        skel.get_bone(i)->a_bone_method();
        const Bone* b = skel.get_bones(i);

        ...
    }
    @endcode
 */
struct SkeletonJoint
{
    SkeletonJoint() { _parent = -1; last_bone_update_sequence = 0; }

    std::shared_ptr<const Bone> _anim_bone;

    // List of children IDs for this bone.
    std::vector<Bone::Id> _children;

    // This joint's parent ID.
    Bone::Id _parent;

    Skeleton_env::Joint_data _joint_data;

    // The bone->update_sequence we last updated bone data.  If this is less than the current
    // bone->update_sequence, we're out of date and need to update_bones_data() when the grid
    // is needed.
    mutable uint64_t last_bone_update_sequence;

    // TODO: this might not be really needed as blending env already stores it
    /// shape of the controller associated to each joint
    /// for the gradient blending operators
    IBL::Ctrl_setup _controller;
};

struct Skeleton {
  friend struct SkeletonImpl;

  // If single_bone is true, the underlying grid will be disabled.  This is slower if the
  // skeleton has many bones, but much faster if it only contains a single bone.
  Skeleton(std::vector<std::shared_ptr<const Bone> > bones, std::vector<Bone::Id> parents, bool single_bone=false);
  ~Skeleton();

  //----------------------------------------------------------------------------
  /// @name Setters
  //----------------------------------------------------------------------------

  void set_joint_controller(int i,
                            const IBL::Ctrl_setup& shape);

  /// @param type The joint type in the enum field in EJoint namespace
  void set_joint_blending(int i, EJoint::Joint_t type);

  /// @param m magnitude for the ith joint. range: [0 1]
  void set_joint_bulge_mag(int i, float m);

  //----------------------------------------------------------------------------
  /// @name Getter
  /// The difference between a joint and a bone must be clear in this section.
  /// A joint is between two bones except for the root joint. The joint frame
  /// used to compute skinning can be different from the bone frame.
  //----------------------------------------------------------------------------

  /// Get the number of joints in the skeleton
  int nb_joints() const { return (int) _joints.size(); }
  std::set<Bone::Id> get_bone_ids() const {
      std::set<Bone::Id> result;
        for(auto &it: _joints)
            result.insert(it.first);
      return result;
  }

  IBL::Ctrl_setup get_joint_controller(Bone::Id i) const;

  /// Get the list of children for the ith bone
  const std::vector<int>& get_sons(Bone::Id i) const { return _joints.at(i)._children;  }

  int parent(int i) const { return _joints.at(i)._parent; }

  // Return true if the joint represents a bone.
  //
  // Root joints don't create bones.
  bool is_bone(int i) const {
      return parent(i) != -1;
  }

  bool is_leaf(int i) const { return _joints.at(i)._children.size() == 0; }

  std::shared_ptr<const Bone> get_bone(Bone::Id i) const{ return _joints.at(i)._anim_bone;  }

  Blending_env::Ctrl_id get_ctrl(int joint) const {
      return _joints.at(joint)._joint_data._ctrl_id;
  }

  float get_joints_bulge_magnitude(Bone::Id i) const {
      return _joints.at(i)._joint_data._bulge_strength;
  }

  Skeleton_env::DBone_id get_bone_didx(Bone::Id i) const;

  /// bone type (whether a primitive is attached to it)
  /// @see Bone Bone_hrbf Bone_ssd Bone_cylinder Bone_precomputed EBone
  EBone::Bone_t bone_type(Bone::Id id_bone) const {
      return _joints.at(id_bone)._anim_bone->get_type();
  }

  /// @return The joint type in the enum field in EJoint namespace
  EJoint::Joint_t joint_blending(Bone::Id i) const {
      return _joints.at(i)._joint_data._blend_type;
  }

  /// Get the id of the skeleton in the skeleton environment
  Skeleton_env::Skel_id get_skel_id() const { return _skel_id; }

  // If bone data is out of date, update bone data.  This is const because it updates internal caches
  // but doesn't change the skeleton's real data; it needs to be called by const users.
  void update_bones_data() const;

private:

  /// Create and initilize a skeleton in the environment Skeleton_env
  void init_skel_env(bool single_bone);

  std::map<Bone::Id, Skeleton_env::Joint_data> get_joints_data() const;

  //----------------------------------------------------------------------------
  /// @name Attributes
  //----------------------------------------------------------------------------

  /// Id of the skeleton in the skeleton environment
  Skeleton_env::Skel_id _skel_id;

  // Maps from bone IDs to joints:
  std::map<Bone::Id, SkeletonJoint> _joints;
};

#endif // SKELETON_HPP__
