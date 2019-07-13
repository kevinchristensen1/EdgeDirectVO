#include "Trajectory.h"
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>


namespace EdgeVO{

void readGroundTruthFile(const std::string filename, std::vector<Pose*>& trajectory, std::vector<double>& timestamps)
{
    std::string line;
    std::ifstream in_stream(filename.c_str());
    while (std::getline(in_stream, line))
    {
        //std::cout << line << std::endl;
        double timestamp;
        double matrixValues[7];
        if(line.empty() || line[0] == '#')
            continue;
        std::stringstream ss(line);
        ss >> std::fixed >> std::setprecision(3) >> timestamp;
        ss >> std::fixed >> std::setprecision(6) >> matrixValues[0] >> matrixValues[1] >> matrixValues[2] 
                        >> matrixValues[4] >> matrixValues[5] >> matrixValues[6] >> matrixValues[3];
        // printf("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", timestamp, matrixValues[0], matrixValues[1], matrixValues[2], 
        //                                                     matrixValues[4], matrixValues[5], matrixValues[6], matrixValues[3]);
        
        
        cv::Mat Rt;
        transAndQuat2Mat(matrixValues, Rt);
        trajectory.push_back(new Pose( Rt ) );
        timestamps.push_back(timestamp);
        //std::cout << std::fixed << std::setprecision(2) << timestamps.back() << std::endl;
        
    }
    in_stream.close();
    return;
}

Trajectory::Trajectory(const std::string filename)
    :m_currentIndex(-1)
{
    readGroundTruthFile(filename, m_groundtruthTrajectory, m_timestamps);
    addPose(*m_groundtruthTrajectory[0]);
    //std::cout << getCurrentPose() << std::endl;
    printf("%.15f" , getCurrentTimestamp() );
}

Trajectory::~Trajectory()
{
    for(size_t i = 0; i < m_trajectory.size(); ++i)
       delete m_trajectory[i];
    
    m_trajectory.clear();
    

    for(size_t i = 0; i < m_groundtruthTrajectory.size(); ++i)
        delete m_groundtruthTrajectory[i];

    m_groundtruthTrajectory.clear();
}

void Trajectory::addPose(Pose& newPose)
{
    ++m_currentIndex;
    m_trajectory.push_back(new Pose(newPose));
}

const Pose& Trajectory::getCurrentPose() const
{
    return *(m_trajectory.back() );
}

const cv::Mat Trajectory::getLastRelativePose() const
{
    if(m_currentIndex > 1)
        return (m_trajectory[m_currentIndex-1]->inversePose() * m_trajectory[m_currentIndex]->getPoseMatrix());
    else
        return cv::Mat::eye(4,4,CV_64FC1);
}

const cv::Mat Trajectory::get2LastRelativePose() const
{
    if(m_currentIndex > 2)
        return se3ExpEigen( se3LogEigen(m_trajectory[m_currentIndex-2]->getPoseMatrix() * m_trajectory[m_currentIndex-1]->inversePose() * m_trajectory[m_currentIndex]->getPoseMatrix()) );
    else
        return cv::Mat::eye(4,4,CV_64FC1);
}

const float Trajectory::getCurrentTimestamp() const
{
    return m_timestamps.back();
}

Pose Trajectory::initializePoseToGroundTruth(double timestamp) const
{
    //const std::vector<const double>::iterator index = upper_bound(m_timestamps.begin(), m_timestamps.end(), timestamp);
    //int index = upper_bound(m_timestamps.begin(), m_timestamps.end(), timestamp) - m_timestamps.begin() - 1;
    //std::cout << timestamp << std::endl;
    //std::cout << (*index) << std::endl;
    //std::cout << *(m_groundtruthTrajectory[index - m_timestamps.begin()-1]) << std::endl;
    return *(m_groundtruthTrajectory[upper_bound(m_timestamps.begin(), m_timestamps.end(), timestamp) - m_timestamps.begin() ]);
}


}
