#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <iostream>
#include <vector>

#include "Frame.h"
#include "Settings.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/ximgproc.hpp>

namespace EdgeVO{
    //General Utility Function

    void readAssocTextfile(std::string filename,
                                  std::vector<std::string>& inputImagePaths,
                                  std::vector<std::string>& inputDepthPaths,
                                  std::vector<std::string>& timestamps);

    inline int getTopPyramidLevel() {return (EdgeVO::Settings::PYRAMID_DEPTH - 1);}
    inline int getBottomPyramidLevel() { return 0; }

class Sequence{
    public:
        explicit Sequence(std::string filename = EdgeVO::Settings::ASSOC_FILE);
        //explicit Sequence(std::string imageDirectory = EdgeVO::Settings::IMAGE_DIRECTORY, std::string depthDirectory = EdgeVO::Settings::DEPTH_DIRECTORY);
        ~Sequence();

        int getFilenamesFromDirectory(std::string dir, std::vector<std::string> &files) const;

        

        Frame* getReferenceFrame();
        Frame* getCurrentFrame();
        
        cv::Mat getCameraMatrix(int level);

        void advanceSequence();
        void makeReferenceFramePyramids();
        void makeCurrentFramePyramids();
        void makeFramePyramids();

        bool sequenceNotFinished();

        int getFrameHeight(int lvl) const;
        int getFrameWidth(int lvl) const;
        double getCurrentTimeStamp() const;
        double getFirstTimeStamp() const;

        cv::Ptr<cv::ximgproc::StructuredEdgeDetection> getSFDetector();
        
        //Display purposes only
        int displayCurrentImage();
        int displayCurrentEdge();
        int displayCurrentDepth();
        void displaySequence();

        void printCameraMatrix(int level);

    private:
        Frame* m_reference;
        Frame* m_current;

        std::vector<std::string> m_depthPaths;
        std::vector<std::string> m_imagePaths;
        std::vector<std::string> m_timestamps;

        std::string m_imageDirectory;
        std::string m_depthDirectory;

        int m_numImageFiles;
        int m_numDepthFiles;

        int m_referenceIndex;
        int m_currentIndex;

        std::vector<cv::Mat> m_cameraMatrix;
        //cv::Mat m_cameraMatrix[EdgeVO::Settings::PYRAMID_DEPTH];
        float fx[EdgeVO::Settings::PYRAMID_DEPTH];
        float fy[EdgeVO::Settings::PYRAMID_DEPTH];
        float cx[EdgeVO::Settings::PYRAMID_DEPTH];
        float cy[EdgeVO::Settings::PYRAMID_DEPTH];

        cv::Ptr<cv::ximgproc::StructuredEdgeDetection> m_sforestDetector;
        //TODO
        Sequence(bool isLive);


};



}
#endif //SEQUENCE_H