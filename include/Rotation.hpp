#include <Eigen/Core>

namespace BubbleTester {
    //! Calculate the rotation matrix for a rotation to the target vector
    /*!
    * Given a target vector, this method computes the rotation matrix
    * to transform the current coordinate system to one in which the
    * first basis vector is aligned with the target vector.
    *
    * Notes:
    * 1. this change of coordinates is not uniquely specified,
    * and may involve a reflection.
    *
    * 2. Right multiplication will take coordinates in the rotated system
    * to coordinates in the unrotated system. Take the transpose for the
    * opposite effect.
    *
    * @param target the target vector to align with
    * @return the rotation matrix
    */
    Eigen::MatrixXd calculate_rotation_to_target(const Eigen::VectorXd& target);
};