#ifndef POSE_H
#define POSE_H
#include <opencv2/core/core.hpp>
#include <iostream>
#include <opencv2/core/utility.hpp>
// #include <Eigen/Dense>
#include <Eigen/Core>
// #include <unsupported/Eigen/MatrixFunctions>
// #include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

namespace EdgeVO{
// transforms from se3 to
void se3Exp(const cv::Mat& input, cv::Mat& toReturn);
cv::Mat se3ExpEigen(const Eigen::Matrix<double, 6 , Eigen::RowMajor>& input);
Eigen::Matrix<double, 6 , Eigen::RowMajor> se3LogEigen(const cv::Mat input);

// 7 vector of format [tx ty tz qx qy qz qw]
void transAndQuat2Mat(const double input[7], cv::Mat& toReturn);

class Pose{
    public:
        Pose();
        Pose(cv::Mat initialValue);
        Pose(const Pose& cp);
        //~Pose();
        cv::Mat getPoseMatrix() const;
        Eigen::Matrix<double,4,4,Eigen::RowMajor> getPoseEigen() const;

        void setPose(const cv::Mat& pose);
        void setIdentityPose();
        void updatePose(const cv::Mat& poseUpdate);
        void updateKeyFramePose(const cv::Mat& keyframe_pose, const cv::Mat& relative_pose);
        cv::Mat inversePose() const;
        Eigen::Matrix<double,4,4,Eigen::RowMajor> inversePoseEigen() const;
        //Assignment operator
        Pose& operator=(const Pose& rhs);

        // Inline operator functions
        inline void operator*=(const Pose &rhs)
        {
            m_pose = m_pose * rhs.m_pose;
        }
        inline void operator*=(const float rhs)
        {
            m_pose = m_pose * rhs;
        }
        inline void operator+=(const Pose &rhs)
        {
            m_pose = m_pose + rhs.m_pose;
        }
        inline void operator+=(const float rhs)
        {
            m_pose = m_pose + rhs;
        }
        inline void operator-=(const Pose &rhs)
        {
            m_pose = m_pose - rhs.m_pose;
        }
        inline void operator-=(const float rhs)
        {
            m_pose = m_pose - rhs;
        }
        inline void operator/=(const float rhs)
        {
            m_pose = m_pose / rhs;
        }

      private:
        cv::Mat m_pose;

};
// Operator definitions
inline const Pose operator*(const Pose& lhs, const Pose& rhs)
{
    Pose p(lhs);
    p *= rhs;
    return p;
}

inline const Pose operator*(const Pose& lhs, const float rhs)
{
    Pose p(lhs);
    p *= rhs;
    return p;
}

inline const Pose operator*(const float lhs, const Pose& rhs)
{
    Pose p(rhs);
    p *= lhs;
    return p;
}

inline const Pose operator/(const Pose& lhs, const float rhs)
{
    Pose p(lhs);
    p /= rhs;
    return p;
}

static inline 
std::ostream &operator<<(std::ostream &out, const Pose &mtx)
{
    return out << mtx.getPoseMatrix();
}

}

#endif //POSE_H