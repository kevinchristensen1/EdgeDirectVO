#include "Frame.h"
#include "Settings.h"
#include <algorithm>
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "CycleTimer.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Sequence.h"
// For fast edge detection using structure forests
#include <opencv2/ximgproc.hpp>

using namespace cv;
namespace EdgeVO{

Frame::Frame()
    :m_image() , m_depthMap()
{}

Frame::Frame(Mat& image)
    :m_image(image) , m_depthMap( Mat() )
{}

Frame::Frame(std::string imagePath, std::string depthPath, Sequence* seq)
    :m_image(cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE)) , m_depthMap(cv::imread(depthPath, cv::ImreadModes::IMREAD_UNCHANGED)) , 
    m_imageName(imagePath), m_depthName(depthPath), m_seq(seq)
{
    //m_image.convertTo(m_image, CV_32FC1);
    //m_depthMap.convertTo(m_depthMap, CV_32FC1, EdgeVO::Settings::PIXEL_TO_METER_SCALE_FACTOR);
    m_pyramidImageUINT.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidImage.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidDepth.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidEdge.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Idx.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Idy.resize(EdgeVO::Settings::PYRAMID_DEPTH);


    m_pyramidImageUINT[0] = m_image.clone(); 
    m_pyramidImage[0] = m_image;
    m_pyramidImage[0].convertTo(m_pyramidImage[0], CV_32FC1);
    m_pyramidDepth[0] = m_depthMap;
    m_pyramidDepth[0].convertTo(m_pyramidDepth[0], CV_32FC1, EdgeVO::Settings::PIXEL_TO_METER_SCALE_FACTOR);
    //m_pyramidImage.push_back(m_image);
#ifdef SFORESTS_EDGES
    m_sforestDetector = m_seq->getSFDetector();//cv::ximgproc::createStructuredEdgeDetection("../model/SForestModel.yml");
    m_pyramidImageSF.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidImageSF[0] = cv::imread(m_imageName,cv::ImreadModes::IMREAD_COLOR);
#else
    m_sforestDetector = nullptr;
#endif
    
}

Frame::Frame(Mat& image, Mat& depthMap)
    :m_image(image) , m_depthMap(depthMap)
{}

/*
Frame::Frame(const Frame& cp)
: m_image( cp.m_image ) , m_depthMap( cp.m_depthMap )
{}
*/
Frame::~Frame()
{
    releaseAllVectors();
}

void Frame::releaseAllVectors()
{
    m_pyramidImage.clear();
    m_pyramid_Idx.clear();
    m_pyramid_Idy.clear();
    m_pyramidDepth.clear();
    m_pyramidMask.clear();
    m_pyramidEdge.clear();
    m_pyramidImageUINT.clear();
    //m_pyramidImageFloat.clear();
}

int Frame::getHeight(int lvl) const
{
    return m_pyramidImage[lvl].rows;
}
int Frame::getWidth(int lvl) const
{
    return m_pyramidImage[lvl].cols;
}
void Frame::printPaths() 
{
    std::cout << m_imageName << std::endl;
    std::cout << m_depthName << std::endl;
}
/*
Frame& Frame::operator=(const Frame& rhs)
{
    if(this == &rhs)
        return *this;
    // copy and swap
    Frame temp(rhs);
    std::swap(*this, temp);
    return *this;
}
*/
Mat& Frame::getImageForDisplayOnly()
{
    return m_pyramidImageUINT[0];
}
Mat& Frame::getEdgeForDisplayOnly()
{
    return m_pyramidEdge[0];
}
Mat& Frame::getDepthForDisplayOnly()
{
    return m_pyramidDepth[0];
}

cv::Mat Frame::getGX(int lvl) const
{
    return m_pyramid_Idx[lvl];
}
cv::Mat Frame::getGY(int lvl) const
{
    return m_pyramid_Idy[lvl];
}

Mat Frame::getImage(int lvl) const
{
    return m_pyramidImage[lvl].clone();
}
cv::Mat Frame::getImageVector(int lvl) const
{
    return (m_pyramidImage[lvl].clone()).reshape(1, m_pyramidImage[lvl].rows * m_pyramidImage[lvl].cols);
}

Mat Frame::getDepthMap(int lvl) const
{
    return (m_pyramidDepth[lvl].clone()).reshape(1, m_pyramidDepth[lvl].rows * m_pyramidDepth[lvl].cols);
}

Mat Frame::getMask(int lvl) const
{
    return m_pyramidMask[lvl].clone();
}
Mat Frame::getEdges(int lvl) const
{
    return (m_pyramidEdge[lvl].clone()).reshape(1, m_pyramidEdge[lvl].rows * m_pyramidEdge[lvl].cols);
}
cv::Mat Frame::getGradientX(int lvl) const
{
    return (m_pyramid_Idx[lvl].clone()).reshape(1, m_pyramid_Idx[lvl].rows * m_pyramid_Idx[lvl].cols);
}
cv::Mat Frame::getGradientY(int lvl) const
{
    return (m_pyramid_Idy[lvl].clone()).reshape(1, m_pyramid_Idy[lvl].rows * m_pyramid_Idy[lvl].cols);
}


void Frame::makePyramids()
{
    createPyramid(m_pyramidImage[0], m_pyramidImage, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_LINEAR);
    cv::buildPyramid(m_pyramidImageUINT[0], m_pyramidImageUINT, EdgeVO::Settings::PYRAMID_BUILD);
    

#ifdef CANNY_EDGES
    // Canny
    createCannyEdgePyramids();
#elif LoG_EDGES
    // LoG
    createLoGEdgePyramids();
#elif SFORESTS_EDGES
    createStructuredForestEdgePyramid();
#elif CONV_BASIN
    createBasinPyramids();
#else
    // Sobel
    createSobelEdgePyramids();
#endif
    createImageGradientPyramids();
    createPyramid(m_pyramidDepth[0], m_pyramidDepth, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    

}

void Frame::createPyramid(cv::Mat& src, std::vector<cv::Mat>& dst, int pyramidSize, int interpolationFlag)
{
    dst.resize(pyramidSize);
    dst[0] = src;
    for(size_t i = 1; i < pyramidSize; ++i)
        cv::resize(dst[i-1], dst[i],cv::Size(0, 0), 0.5, 0.5, interpolationFlag);
    
    
}

void Frame::createImageGradientPyramids()
{
    int one(1);
    int zero(0);
    double scale = 0.5;

    calcGradientX(m_pyramidImage[0], m_pyramid_Idx[0]);
    calcGradientY(m_pyramidImage[0], m_pyramid_Idy[0]);
   
    createPyramid(m_pyramid_Idx[0], m_pyramid_Idx, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    createPyramid(m_pyramid_Idy[0], m_pyramid_Idy, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
}

void Frame::calcGX(cv::Mat &src)
{
    m_GX.resize(src.rows, src.cols);
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(x == 0)
                m_GX(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x));
            else if(x == src.cols-1)
                m_GX(y,x) = (src.at<float>(y,x) - src.at<float>(y,x-1));
            else
                m_GX(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x-1))*0.5;
        }
    }
}

void Frame::calcGradientX(cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f).clone();
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(x == 0)
                dst.at<float>(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x));
            else if(x == src.cols-1)
                dst.at<float>(y,x) = (src.at<float>(y,x) - src.at<float>(y,x-1));
            else
                dst.at<float>(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x-1))*0.5;
        }
    }
}
void Frame::calcGradientY(cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f ).clone();
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(y == 0)
                dst.at<float>(y,x) = (src.at<float>(y+1,x) - src.at<float>(y,x));
            else if(y == src.rows-1)
                dst.at<float>(y,x) = (src.at<float>(y,x) - src.at<float>(y-1,x));
            else
                dst.at<float>(y,x) = (src.at<float>(y+1,x) - src.at<float>(y-1,x))*0.5;
        }
    }
}
void Frame::createCannyEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_thresh; //not used
        //cv::GaussianBlur( m_pyramidImageUINT[i], m_pyramidImageUINT[i], Size(3,3), EdgeVO::Settings::SIGMA);
  /// Canny detector
        float upperThreshold = cv::threshold(m_pyramidImageUINT[i], img_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        float lowerThresh = EdgeVO::Settings::CANNY_RATIO * upperThreshold;
        Canny(m_pyramidImageUINT[i], m_pyramidEdge[i], lowerThresh, upperThreshold, 3, true);
    }
    //void Canny(InputArray image, OutputArray edges, float threshold1, float threshold2, int apertureSize=3, bool L2gradient=false )
}

void Frame::createLoGEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_dest; 
        cv::GaussianBlur( m_pyramidImageUINT[i], img_dest, Size(3,3), 0, 0, cv::BORDER_DEFAULT );
        cv::Laplacian( img_dest, img_dest, CV_8UC1, 3, 1., 0, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( img_dest, img_dest );
        cv::threshold(img_dest, m_pyramidEdge[i], 25, 255, cv::THRESH_BINARY);
    }

}
void Frame::createSobelEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        cv::Mat grad_x, grad_y;
        cv::Mat grad;
        /// x Gradient
        Sobel( m_pyramidImageUINT[i], grad_x, CV_16S, 1, 0, 3, 1., 0, cv::BORDER_DEFAULT );
        convertScaleAbs( grad_x, grad_x );
        /// y Gradient
        Sobel( m_pyramidImageUINT[i], grad_y, CV_16S, 0, 1, 3, 1., 0, cv::BORDER_DEFAULT );
        convertScaleAbs( grad_y, grad_y );
        addWeighted( grad_x, 0.5, grad_y, 0.5, 0, grad );
        double max;
        double min;
        cv::minMaxLoc(grad, &min, &max);
        cv::threshold(grad/max, m_pyramidEdge[i], 0.95, 255, cv::THRESH_BINARY);
    }

}
void Frame::createStructuredForestEdgePyramid()
{
    cv::buildPyramid(m_pyramidImageSF[0], m_pyramidImageSF, EdgeVO::Settings::PYRAMID_BUILD);
    for(size_t i = 0; i < m_pyramidImageSF.size(); ++i)
    {
        Mat image = m_pyramidImageSF[i].clone();
        image.convertTo(image, CV_32FC3, 1./255.0);
        cv::Mat edges(image.size(), image.type());
        m_sforestDetector->detectEdges(image, edges );
        cv::threshold(edges, m_pyramidEdge[i], 0.15, 255, cv::THRESH_BINARY);
     
    }
        

}
void Frame::createBasinPyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_thresh; //not used
        cv::GaussianBlur( m_pyramidImageUINT[i], m_pyramidImageUINT[i], Size(3,3), EdgeVO::Settings::SIGMA);
  /// Canny detector
        float upperThreshold = cv::threshold(m_pyramidImageUINT[i], img_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        float lowerThresh = EdgeVO::Settings::CANNY_RATIO * upperThreshold;
        Canny(m_pyramidImageUINT[i], m_pyramidEdge[i], lowerThresh, upperThreshold, 3, true);
    }
    m_pyramidEdge[m_pyramidEdge.size()-1] = cv::Mat::ones(m_pyramidImageUINT[m_pyramidEdge.size()-1].rows, m_pyramidImageUINT[m_pyramidEdge.size()-1].cols, CV_8UC1);
}

bool Frame::hasDepthMap()
{
    return !(m_depthMap.empty() );

}

void Frame::setDepthMap(Mat& depthMap)
{
    if(!hasDepthMap())
        m_depthMap = depthMap;
    // Otherwise do nothing
}

}