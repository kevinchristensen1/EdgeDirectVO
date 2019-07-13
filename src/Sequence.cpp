#include "Sequence.h"
#include "Settings.h"
#include "CycleTimer.h"

#include <fstream>
#include <dirent.h>
#include <algorithm>

#include <utility>
#include <functional>

#include <string>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv2/ximgproc.hpp>

#include <unsupported/Eigen/MatrixFunctions>

namespace EdgeVO{

void readAssocTextfile(std::string filename,
                       std::vector<std::string>& inputRGBPaths,
                       std::vector<std::string>& inputDepthPaths,
                       std::vector<std::string>& timestamps) 
{
  std::string line;
  std::ifstream in_stream(filename.c_str());
  while (!in_stream.eof()) {
    std::getline(in_stream, line);
    std::stringstream ss(line);
    std::string buf;
    int count = 0;
    while (ss >> buf) {
      count++;
      if (count == 1)
      {
        timestamps.push_back(buf);
      }
      if (count == 2) {
        inputRGBPaths.push_back(EdgeVO::Settings::DATASET_DIRECTORY + buf);
      } else if (count == 4) {
        inputDepthPaths.push_back(EdgeVO::Settings::DATASET_DIRECTORY + buf);
      }
    }
  }
  in_stream.close();
}


// std::string &ltrim(std::string &s) {
//         s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
//         return s;
// }
// std::string &rtrim(std::string &s) {
//         s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
//         return s;
// }
// std::string &trim(std::string &s) {
//         return ltrim(rtrim(s));
// }



Sequence::Sequence(std::string filename)
    : m_referenceIndex(0) , m_currentIndex(1)
{
    readAssocTextfile(filename, m_imagePaths, m_depthPaths, m_timestamps);
    
    m_numDepthFiles = m_depthPaths.size();
    m_numImageFiles = m_imagePaths.size();
    // add assertion that they are the same size
#ifdef SFORESTS_EDGES
    m_sforestDetector = cv::ximgproc::createStructuredEdgeDetection("../model/SForestModel.yml");
#endif
    //cv::Mat a = cv::imread(m_imagePaths[m_referenceIndex], cv::ImreadModes::IMREAD_GRAYSCALE);
    m_reference = new Frame(m_imagePaths[m_referenceIndex], m_depthPaths[m_referenceIndex], this);
    m_current = new Frame(m_imagePaths[m_currentIndex], m_depthPaths[m_currentIndex], this);
    //++m_referenceIndex;
   // ++m_currentIndex;

    // Build Camera Matrix
    for(int i = 0; i < EdgeVO::Settings::PYRAMID_DEPTH ; ++i)
    {
        float denominator(std::pow(2., i));
        fx[i] = EdgeVO::Settings::FX / denominator;
        fy[i] = EdgeVO::Settings::FY / denominator;
        // if(i != 0)
        // {
        //     cx[i] = (EdgeVO::Settings::CX-0.5f) / denominator + 0.5f; 
        //     cy[i] = (EdgeVO::Settings::CY-0.5f) / denominator + 0.5f;
        // }
        // else
        // {
        //     cx[i] = EdgeVO::Settings::CX / denominator; 
        //     cy[i] = EdgeVO::Settings::CY / denominator;
        // }
        
        
        cx[i] = EdgeVO::Settings::CX / denominator; 
        cy[i] = EdgeVO::Settings::CY / denominator;

        float matrixValues[] = {fx[i], 0.f, cx[i], 0.f, fy[i], cy[i], 0.f, 0.f, 1.f};
        m_cameraMatrix.push_back( cv::Mat(3, 3, CV_32FC1, matrixValues).clone() ); //Mat is implemented as a smart ptr
    }
#ifdef DISPLAY_SEQUENCE
    cv::namedWindow( EdgeVO::Settings::DISPLAY_SEQUENCE_WINDOW , cv::WINDOW_AUTOSIZE );
    cv::namedWindow( EdgeVO::Settings::DISPLAY_EDGE_WINDOW , cv::WINDOW_AUTOSIZE );
    cv::namedWindow( EdgeVO::Settings::DISPLAY_DEPTH_WINDOW , cv::WINDOW_AUTOSIZE );
#endif //DISPLAY_SEQUENCE

}

/*
Sequence::Sequence(std::string imageDirectory, std::string depthDirectory)
    :m_imageDirectory(imageDirectory) , m_depthDirectory(depthDirectory) , 
     m_referenceIndex(0) , m_currentIndex(1)
{
    m_numImageFiles = this->getFilenamesFromDirectory(m_imageDirectory, m_imagePaths);
    m_numDepthFiles = this->getFilenamesFromDirectory(m_depthDirectory, m_depthPaths);

    if(m_numImageFiles >= 0)
		printf("[Sequence] Found %d image files in folder [ %s ]!\n", m_numImageFiles, m_imageDirectory.c_str());
    if(m_numDepthFiles >= 0)
        printf("[Sequence] Found %d depth files in folder [ %s ]!\n", m_numDepthFiles, m_depthDirectory.c_str());
    
}
*/

Sequence::~Sequence()
{
    cv::destroyAllWindows();
    m_depthPaths.clear();
    m_imagePaths.clear();
    m_timestamps.clear();

    delete m_reference;
    delete m_current;
}

// TODO Constructor for live frame feed
Sequence::Sequence(bool isLive)
{}

cv::Mat Sequence::getCameraMatrix(int level)
{
    return m_cameraMatrix[level].clone();
}

Frame* Sequence::getReferenceFrame()
{
    return m_reference;
}

Frame* Sequence::getCurrentFrame()
{
    return m_current;
}

void Sequence::printCameraMatrix(int level)
{
    if(level > EdgeVO::Settings::PYRAMID_OUT_OF_BOUNDS)
    {
        std::cout << "Error" << std::endl;
    }
        
    std::cout << m_cameraMatrix[level] << std::endl;
}

void Sequence::advanceSequence()
{
    ++m_currentIndex;
    ++m_referenceIndex;
    if (m_referenceIndex % EdgeVO::Settings::KEYFRAME_INTERVAL == 0)
    {
        delete m_reference;
        m_reference = m_current;
    }
    else
    {
        delete m_current;
    }
    
    m_current = new Frame(m_imagePaths[m_currentIndex], m_depthPaths[m_currentIndex], this);

}

void Sequence::makeCurrentFramePyramids()
{
    m_current->makePyramids();
    return;
}

void Sequence::makeReferenceFramePyramids()
{
    m_reference->makePyramids();
    return;
}
void Sequence::makeFramePyramids()
{
    makeCurrentFramePyramids();
    makeReferenceFramePyramids();
    return;
}

int Sequence::getFrameHeight(int lvl) const
{
    return m_reference->getHeight(lvl);
}
int Sequence::getFrameWidth(int lvl) const
{
    return m_reference->getWidth(lvl);
}

int Sequence::getFilenamesFromDirectory(std::string dir, std::vector<std::string> &files) const
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
        return -1;

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != "..")
    		files.push_back(name);
    }
    closedir(dp);
    // Sort vector by name (which is also timestamp)
    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(size_t i = 0; i<files.size(); i++)
	{
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}

double Sequence::getFirstTimeStamp() const
{
    //printf("first Timestamp: %.10f", std::stod( m_timestamps[0] ));
    return std::stod( m_timestamps[0] );
}

double Sequence::getCurrentTimeStamp() const
{
    //std::cout << m_currentIndex << std::endl << m_timestamps[m_currentIndex] << std::endl;
    return std::stod( m_timestamps[m_currentIndex] );
}

bool Sequence::sequenceNotFinished()
{
    return (m_currentIndex < m_numDepthFiles);
}

int Sequence::displayCurrentImage()
{
    cv::imshow( EdgeVO::Settings::DISPLAY_SEQUENCE_WINDOW , m_current->getImageForDisplayOnly() ); 
    return cv::waitKey(1);
}
int Sequence::displayCurrentEdge()
{
    cv::imshow( EdgeVO::Settings::DISPLAY_EDGE_WINDOW , m_current->getEdgeForDisplayOnly() ); 
    return cv::waitKey(1);

}
int Sequence::displayCurrentDepth()
{
    cv::imshow( EdgeVO::Settings::DISPLAY_DEPTH_WINDOW , m_current->getDepthForDisplayOnly() ); 
    return cv::waitKey(1);

}

cv::Ptr<cv::ximgproc::StructuredEdgeDetection> Sequence::getSFDetector()
{
    return m_sforestDetector;
}

void Sequence::displaySequence()
{
    //cv::namedWindow( EdgeVO::Settings::DISPLAY_SEQUENCE_WINDOW , cv::WINDOW_AUTOSIZE );

    for(size_t i = 0; i < m_numImageFiles; i++)
    {
        cv::Mat imageDist = cv::imread(m_imagePaths[i], cv::ImreadModes::IMREAD_UNCHANGED);
		float startTime = EdgeVO::CycleTimer::currentSeconds();

		cv::imshow( EdgeVO::Settings::DISPLAY_SEQUENCE_WINDOW , imageDist );  
		int keyPressed = cv::waitKey(1);
		float endTime = EdgeVO::CycleTimer::currentSeconds();

        if(keyPressed == EdgeVO::Settings::TERMINATE_DISPLAY_KEY) 
        {
            printf("Display Terminated by User\n");
            break;
        }
            
		printf("[Display Images]:\t\t[%.3f] ms\n", (endTime - startTime) * 1000);
    }

    cv::destroyWindow(EdgeVO::Settings::DISPLAY_SEQUENCE_WINDOW);
}


}