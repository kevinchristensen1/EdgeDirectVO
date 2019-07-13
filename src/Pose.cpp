#include "Pose.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/eigen.hpp>
#include "Settings.h"
#include <math.h>


namespace EdgeVO{
    //using namespace cv;

Pose::Pose()
    :m_pose(cv::Mat::eye(4,4,CV_64FC1))
{}

Pose::Pose(cv::Mat initialValue)
    :m_pose(initialValue.clone())
{}

Pose::Pose(const Pose& cp)
    :m_pose(cp.m_pose.clone())
{}

void Pose::setIdentityPose()
{
    m_pose = cv::Mat::eye(4,4,CV_64FC1);
}

void Pose::setPose(const cv::Mat& pose)
{
    m_pose = pose.clone();
}
void Pose::updatePose(const cv::Mat& poseUpdate)
{
    m_pose = poseUpdate * m_pose;
    return; 
}

void Pose::updateKeyFramePose(const cv::Mat& keyframe_pose, const cv::Mat& relative_pose)
{
    m_pose = keyframe_pose * relative_pose;
    return;
}
 
cv::Mat Pose::getPoseMatrix() const
{
    // deep copy
    return m_pose.clone();
}

Eigen::Matrix<double,4,4,Eigen::RowMajor> Pose::getPoseEigen() const
{
    Eigen::Matrix<double,4,4, Eigen::RowMajor> pose;
    cv::cv2eigen(m_pose, pose);
    return pose;
}

cv::Mat Pose::inversePose() const
{
    Eigen::Matrix<double,4,4, Eigen::RowMajor> invPose;
    cv::Mat toReturn(4,4, CV_64FC1);

    cv::cv2eigen(m_pose.clone(),invPose);
    invPose.block<3,3>(0,0).transposeInPlace();
    invPose.block<3,1>(0,3) = - invPose.block<3,3>(0,0) * invPose.block<3,1>(0,3);

    cv::eigen2cv(invPose, toReturn);
    return toReturn.clone();
}

Eigen::Matrix<double,4,4, Eigen::RowMajor>  Pose::inversePoseEigen() const
{
    Eigen::Matrix<double,4,4,Eigen::RowMajor> invPose;
    cv::cv2eigen(m_pose.clone(),invPose);
    invPose.block<3,3>(0,0).transposeInPlace();
    invPose.block<3,1>(0,3) = - invPose.block<3,3>(0,0) * invPose.block<3,1>(0,3);
    return invPose;
}

Pose& Pose::operator=(const Pose& rhs)
{
    if(this == &rhs)
        return *this;
    
    m_pose = rhs.getPoseMatrix(); //makes a clone
    return *this;

}

void se3Exp(const cv::Mat& input, cv::Mat& toReturn)
{
    // Size of input must be [6 x 1] or [1 x 6]
    CV_Assert( (input.size() == cv::Size(6,1) || input.size() == cv::Size(1,6) ) && input.type() == CV_32FC1);

    Eigen::Matrix<float,4,4,Eigen::RowMajor> lie;
    lie << 0.f,                       -input.at<float>(5), input.at<float>(4),  input.at<float>(0),
           input.at<float>(5),      0.f,                   -input.at<float>(3), input.at<float>(1),
           -input.at<float>(4),     input.at<float>(3),    0.f,                 input.at<float>(2),
           0.f,                       0.f,                   0.f,                   0.f;

    // Exponentiate
    Eigen::Matrix<float,4,4,Eigen::RowMajor> Rt( lie.exp() );

    cv::eigen2cv(Rt, toReturn);
    
    return;
}


cv::Mat se3ExpEigen(const Eigen::Matrix<double, 6 , Eigen::RowMajor>& input)
{
    // Size of input must be [6 x 1] or [1 x 6]
    //CV_Assert( (input.size() == cv::Size(6,1) || input.size() == cv::Size(1,6) ) && input.type() == CV_32FC1);

    Eigen::Matrix<double,4,4,Eigen::RowMajor> lie;
    lie << 0.,           -input[5],  input[4],   input[0],
           input[5],      0.,        -input[3],  input[1],
           -input[4],     input[3],   0.,        input[2],
           0.,           0.,        0.,        0.;

    // Exponentiate
    Eigen::Matrix<double,4,4, Eigen::RowMajor> Rt( lie.exp() );
    cv::Mat toReturn;
    cv::eigen2cv(Rt, toReturn);
    return toReturn.clone();
}

Eigen::Matrix<double, 6 , Eigen::RowMajor> se3LogEigen(const cv::Mat input)
{
    // Size of input must be [6 x 1] or [1 x 6]
    //CV_Assert( (input.size() == cv::Size(6,1) || input.size() == cv::Size(1,6) ) && input.type() == CV_32FC1);
    Eigen::Matrix<double, 6 , Eigen::RowMajor> logMap;
    Eigen::Matrix<double,4,4, Eigen::RowMajor> Rt;
    cv::cv2eigen(input, Rt);
    Eigen::Matrix<double,3,3, Eigen::RowMajor> R = Rt.block<3,3>(0,0);
    Eigen::Matrix<double,3, Eigen::RowMajor> t = Rt.block<3,1>(0,3);
    Eigen::Matrix<double,3,3, Eigen::RowMajor> A = 0.5*(R - R.transpose());
    double s = sqrt( A(2,1)*A(2,1) + A(0,2) * A(0,2) + A(1,0) * A(1,0) );
    double c = 0.5*(R.trace() - 1);
    if(s < EdgeVO::Settings::EPSILON & (c-1.) < EdgeVO::Settings::EPSILON)
    {
        logMap << t , 0. , 0., 0.;
        return logMap;
    }
    else
    {
        double theta = atan2(s,c);
        Eigen::Matrix<double,3, Eigen::RowMajor> r;
        r << A(2,1) , A(0,2) , A(1,0);
        r = (theta/s)*r;
        Eigen::Matrix<double,3,3, Eigen::RowMajor> V;
        Eigen::Matrix<double,3,3, Eigen::RowMajor> wx;
        Eigen::Matrix<double,3, Eigen::RowMajor> tp;
        wx << 0. , -r[2], r[1],
              r[2], 0,    -r[0],
              -r[1], r[0], 0.;
        V.setIdentity();
        V = V + ((1.-c)/ pow(theta,2.)) * wx  + ((theta- s)/pow(theta,3.))*wx*wx;
        tp = V.ldlt().solve(t);
        logMap.head<3>() = tp;
        logMap.tail<3>() = r;
        return logMap;
    }
    
}

void transAndQuat2Mat(const double input[7], cv::Mat& toReturn)
{
    //Eigen::Matrix<float,4,1> quat();
    Eigen::Quaternion<double> q(input[3], input[4], input[5], input[6]);
    Eigen::Matrix<double,4,4,Eigen::RowMajor> Rt = Eigen::Matrix4d::Zero();
    Rt.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    Rt(0,3) = input[0];
    Rt(1,3) = input[1];
    Rt(2,3) = input[2];
    Rt(3,3) = 1.f;
    cv::eigen2cv(Rt, toReturn);
    return;

}

}