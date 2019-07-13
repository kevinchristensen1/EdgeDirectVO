#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <opencv2/core/core.hpp>
#include "Pose.h"
#include "Settings.h"

namespace EdgeVO{
    void readGroundTruthFile(const std::string filename, std::vector<Pose*>& trajectory, std::vector<double>& timestamps);

class Trajectory{
    public:
        Trajectory(const std::string filename = EdgeVO::Settings::GROUNDTRUTH_FILE);
        Trajectory(const Trajectory& cp);
        ~Trajectory();
        Trajectory& operator=(const Trajectory& rhs);

        void addPose(Pose& newPose);

        const cv::Mat getLastRelativePose() const;

        const cv::Mat get2LastRelativePose() const;
        
        const Pose& getCurrentPose() const;
        const float getCurrentTimestamp() const;

        Pose initializePoseToGroundTruth(double timestamp) const;


    private:
        std::vector<Pose*> m_trajectory;
        std::vector<Pose*> m_groundtruthTrajectory;
        std::vector<double> m_timestamps;
        Pose m_currentPose;
        int m_currentIndex;


};
}

#endif //TRAJECTORY_H