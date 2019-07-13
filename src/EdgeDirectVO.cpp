#include "EdgeDirectVO.h"
#include "CycleTimer.h"
#include <algorithm>
#include <utility>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pose.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stdio.h> 
#include <stdlib.h> 
#include <random>
#include <iterator>
#include <algorithm>


namespace EdgeVO{
    using namespace cv;
EdgeDirectVO::EdgeDirectVO()
    :m_sequence(EdgeVO::Settings::ASSOC_FILE) , m_trajectory() , 
     m_lambda(0.)
{
    int length = m_sequence.getFrameHeight( getBottomPyramidLevel() ) * m_sequence.getFrameWidth( getBottomPyramidLevel() );
    
    m_X3DVector.resize(EdgeVO::Settings::PYRAMID_DEPTH); // Vector for each pyramid level
    for(size_t i = 0; i < m_X3DVector.size(); ++i)
        m_X3DVector[i].resize(length / std::pow(4, i) , Eigen::NoChange); //3 Vector for each pyramid for each image pixel

    m_X3D.resize(length, Eigen::NoChange);
    m_warpedX.resize(length);
    m_warpedY.resize(length);
    m_warpedZ.resize(length);
    m_gx.resize(length);
    m_gxFinal.resize(length);
    m_gy.resize(length);
    m_gyFinal.resize(length);
    m_im1.resize(length);
    m_im1Final.resize(length);
    m_im2Final.resize(length);
    m_ZFinal.resize(length);
    m_Z.resize(length);
    m_edgeMask.resize(length);

    m_outputFile.open(EdgeVO::Settings::RESULTS_FILE);

}

EdgeDirectVO::EdgeDirectVO(const EdgeDirectVO& cp)
    :m_sequence(EdgeVO::Settings::ASSOC_FILE)
{}

EdgeDirectVO::~EdgeDirectVO()
{
    m_outputFile.close();
}

EdgeDirectVO& EdgeDirectVO::operator=(const EdgeDirectVO& rhs)
{
    if(this == &rhs)
        return *this;
    
    EdgeDirectVO temp(rhs);
    std::swap(*this, temp);
    return *this;

}

void EdgeDirectVO::runEdgeDirectVO()
{
    //Start timer for stats
    m_statistics.start();
    //Make Pyramid for Reference frame
    m_sequence.makeReferenceFramePyramids();
    // Run for entire sequence

    //Prepare some vectors
    prepare3DPoints();
    //Init camera_pose with ground truth trajectory to make comparison easy
    Pose camera_pose = m_trajectory.initializePoseToGroundTruth(m_sequence.getFirstTimeStamp());
    Pose keyframe_pose = camera_pose;
    // relative_pose intiialized to identity matrix
    Pose relative_pose;

    // Start clock timer
    
    outputPose(camera_pose, m_sequence.getFirstTimeStamp());
    m_statistics.addStartTime((float) EdgeVO::CycleTimer::currentSeconds());
    for (size_t n = 0; m_sequence.sequenceNotFinished(); ++n)
    {
        std::cout << std::endl << camera_pose << std::endl;

#ifdef DISPLAY_SEQUENCE
        //We re-use current frame for reference frame info
        m_sequence.makeCurrentFramePyramids();

        //Display images
        int keyPressed1 = m_sequence.displayCurrentImage();
        int keyPressed2 = m_sequence.displayCurrentEdge();
        int keyPressed3 = m_sequence.displayCurrentDepth();
        if(keyPressed1 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY 
            || keyPressed2 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY
            || keyPressed3 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY) 
        {
            terminationRequested();
            break;
        }
        //Start algorithm timer for each iteration
        float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
#else
        //Start algorithm timer for each iteration
        float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
        m_sequence.makeCurrentFramePyramids();
#endif //DISPLAY_SEQUENCE

        if( n % EdgeVO::Settings::KEYFRAME_INTERVAL == 0 )
        {
            keyframe_pose = camera_pose;
            relative_pose.setIdentityPose();
        }
        //Constant motion assumption
        relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.getLastRelativePose());
        relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));

        //Constant acc. assumption
        //relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.get2LastRelativePose());
        //relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));
        
        // For each image pyramid level, starting at the top, going down
        for (int lvl = getTopPyramidLevel(); lvl >= getBottomPyramidLevel(); --lvl)
        {
            
            const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
            prepareVectors(lvl);
            
            //make3DPoints(cameraMatrix, lvl);

            float lambda = 0.f;
            float error_last = EdgeVO::Settings::INF_F;
            float error = error_last;
            for(int i = 0; i < EdgeVO::Settings::MAX_ITERATIONS_PER_PYRAMID[ lvl ]; ++i)
            {
                error_last = error;
                error = warpAndProject(relative_pose.inversePoseEigen(), lvl);
                // Levenberg-Marquardt
                if( error < error_last)
                {
                    // Update relative pose
                    Eigen::Matrix<double, 6 , Eigen::RowMajor> del;
                    solveSystemOfEquations(lambda, lvl, del);
                    //std::cout << del << std::endl;

                    
                    if( (del.segment<3>(0)).dot(del.segment<3>(0)) < EdgeVO::Settings::MIN_TRANSLATION_UPDATE & 
                        (del.segment<3>(3)).dot(del.segment<3>(3)) < EdgeVO::Settings::MIN_ROTATION_UPDATE    )
                        break;

                    cv::Mat delMat = se3ExpEigen(del);
                    relative_pose.updatePose( delMat );

                    //Update lambda
                    if(lambda <= EdgeVO::Settings::LAMBDA_MAX)
                        lambda = EdgeVO::Settings::LAMBDA_MIN;
                    else
                        lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
                }
                else
                {
                    if(lambda == EdgeVO::Settings::LAMBDA_MIN)
                        lambda = EdgeVO::Settings::LAMBDA_MAX;
                    else
                        lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
                }
            }
        }
        camera_pose.updateKeyFramePose(keyframe_pose.getPoseMatrix(), relative_pose.getPoseMatrix());
        outputPose(camera_pose, m_sequence.getCurrentTimeStamp());
        //At end, update sequence for next image pair
        float endTime = (float) EdgeVO::CycleTimer::currentSeconds();
        m_trajectory.addPose(camera_pose);
                
        // Don't time past this part (reading from disk)
        m_sequence.advanceSequence();
        m_statistics.addDurationForFrame(startTime, endTime);
        m_statistics.addCurrentTime((float) EdgeVO::CycleTimer::currentSeconds());
        m_statistics.printStatistics();
        
    }
    // End algorithm level timer
    m_statistics.end();
    return;
}

void EdgeDirectVO::prepareVectors(int lvl)
{
    cv2eigen(m_sequence.getReferenceFrame()->getDepthMap(lvl), m_Z);
    cv2eigen(m_sequence.getCurrentFrame()->getEdges(lvl), m_edgeMask);
    cv2eigen(m_sequence.getReferenceFrame()->getImageVector(lvl), m_im1);
    cv2eigen(m_sequence.getCurrentFrame()->getImageVector(lvl), m_im2);
    cv2eigen(m_sequence.getCurrentFrame()->getGradientX(lvl), m_gx);
    cv2eigen(m_sequence.getCurrentFrame()->getGradientY(lvl), m_gy);
    
    size_t numElements;
////////////////////////////////////////////////////////////
// REGULAR_DIRECT_VO
////////////////////////////////////////////////////////////
#ifdef REGULAR_DIRECT_VO
    m_edgeMask = (m_edgeMask.array() == 0).select(1, m_edgeMask);
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    m_edgeMask = (m_gx.array()*m_gx.array() + m_gy.array()*m_gy.array() <= EdgeVO::Settings::MIN_GRADIENT_THRESH).select(0, m_edgeMask);
#elif REGULAR_DIRECT_VO_SUBSET
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
    m_edgeMask = (m_edgeMask.array() == 0).select(1, m_edgeMask);
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);

#else
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    //m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    //size_t numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
#endif //REGULAR_DIRECT_VO

////////////////////////////////////////////////////////////
// EDGEVO_SUBSET_POINTS
////////////////////////////////////////////////////////////
#ifdef EDGEVO_SUBSET_POINTS
    //numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
    //size_t numElements = (m_edgeMask.array() != 0).count() < EdgeVO::Settings::NUMBER_POINTS ? (m_edgeMask.array() != 0).count() : EdgeVO::Settings::NUMBER_POINTS;
    std::vector<size_t> indices, randSample;
    m_im1Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    m_X3D.resize(numElements ,Eigen::NoChange);
    m_finalMask.resize(numElements);

    //size_t idx = 0;
    for(int i = 0; i < m_edgeMask.rows(); ++i)
    {
        if(m_edgeMask[i] != 0)
        {
            indices.push_back(i);
        }
    }
    std::sample(indices.begin(), indices.end(), std::back_inserter(randSample),
                numElements, std::mt19937{std::random_device{}()});
    
    //size_t idx = 0;
    for(int i = 0; i < randSample.size(); ++i)
    {
        m_im1Final[i] = m_im1[randSample[i]];
        m_ZFinal[i] = m_Z[randSample[i]];
        m_X3D.row(i) = (m_X3DVector[lvl].row(randSample[i])).array() * m_Z[randSample[i]];
        m_finalMask[i] = m_edgeMask[randSample[i]];    
    }


#else
////////////////////////////////////////////////////////////
// Edge Direct VO
////////////////////////////////////////////////////////////
    numElements = (m_edgeMask.array() != 0).count();
    
    m_im1Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    m_X3D.resize(numElements ,Eigen::NoChange);
    m_finalMask.resize(numElements);
    size_t idx = 0;
    for(int i = 0; i < m_edgeMask.rows(); ++i)
    {
        if(m_edgeMask[i] != 0)
        {
            m_im1Final[idx] = m_im1[i];
            m_ZFinal[idx] = m_Z[i];
            m_X3D.row(idx) = (m_X3DVector[lvl].row(i)).array() * m_Z[i];
            m_finalMask[idx] = m_edgeMask[i];
            ++idx;
        }
    }

#endif //EDGEVO_SUBSET_POINTS
////////////////////////////////////////////////////////////
    m_Z.resize(numElements);
    m_Z = m_ZFinal;
    m_edgeMask.resize(numElements);
    m_edgeMask = m_finalMask;
    
}

void EdgeDirectVO::make3DPoints(const cv::Mat& cameraMatrix, int lvl)
{
    m_X3D = m_X3DVector[lvl].array() * m_Z.replicate(1, m_X3DVector[lvl].cols() ).array();
}

float EdgeDirectVO::warpAndProject(const Eigen::Matrix<double,4,4>& invPose, int lvl)
{
    Eigen::Matrix<float,3,3> R = (invPose.block<3,3>(0,0)).cast<float>() ;
    Eigen::Matrix<float,3,1> t = (invPose.block<3,1>(0,3)).cast<float>() ;
    //std::cout << R << std::endl << t << std::endl;
    //std::cout << "Cols: " << m_X3D[lvl].cols() << "Rows: " << m_X3D[lvl].rows() << std::endl;
    
    m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
    m_newX3D = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows() );

    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);
    //std::cout << cy << std::endl;
    //exit(1);
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);

    m_warpedX.resize(m_X3D.rows());
    m_warpedY.resize(m_X3D.rows());

    m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array() ) + cx;
    //m_warpedX.array() += cx;
    m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array() ) + cy;
    //m_warpedY.array() += cy;

    // (R.array() < s).select(P,Q );  // (R < s ? P : Q)
    //std::cout << newX3D.rows() << std::endl;
    //std::cout << m_finalMask.rows() << std::endl;

    // Check both Z 3D points are >0
    //m_finalMask = m_edgeMask;

    m_finalMask = m_edgeMask;

    m_finalMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_newX3D.row(2).transpose().array() > EdgeVO::Settings::MAX_Z_DEPTH).select(0, m_finalMask);

    //m_finalMask = (m_newX3D.row(2).transpose().array() > 10.f).select(0, m_finalMask);
    m_finalMask = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_X3D.col(2).array() > 10.f).select(0, m_finalMask);
    m_finalMask = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMask, 0);
    m_finalMask = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMask, 0);
    
    // Check new projected x coordinates are: 0 <= x < w-1
    m_finalMask = (m_warpedX.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array() >= w-2).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array().isFinite()).select(m_finalMask, 0);
    // Check new projected x coordinates are: 0 <= y < h-1
    m_finalMask = (m_warpedY.array() >= h-2).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array().isFinite()).select(m_finalMask, 0);
    

// If we want every point, save some computation time- see the #else
////////////////////////////////////////////////////////////
#ifdef EDGEVO_SUBSET_POINTS_EXACT
    size_t numElements = (m_finalMask.array() != 0).count() < EdgeVO::Settings::NUMBER_POINTS ? (m_finalMask.array() != 0).count() : EdgeVO::Settings::NUMBER_POINTS;

    //size_t numElements = (m_finalMask.array() != 0).count();
    m_gxFinal.resize(numElements);
    m_gyFinal.resize(numElements);
    m_im1.resize(numElements);
    m_im2Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    std::vector<size_t> indices, randSample;

    //size_t idx = 0;
    for(int i = 0; i < m_finalMask.rows(); ++i)
    {
        if(m_finalMask[i] != 0)
        {
            indices.push_back(i);
        }
    }
    std::sample(indices.begin(), indices.end(), std::back_inserter(randSample),
                numElements, std::mt19937{std::random_device{}()});
    
    size_t idx = 0;
    for(int i = 0; i < randSample.size(); ++i)
    {
        m_gxFinal[i]  = interpolateVector( m_gx, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_gyFinal[i]  = interpolateVector( m_gy, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_im1[i] = m_im1Final[randSample[i]];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
        m_im2Final[i] = interpolateVector(m_im2, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_XFinal[i] = m_newX3D(0,randSample[i]);
        m_YFinal[i] = m_newX3D(1,randSample[i]);
        m_ZFinal[i] = m_newX3D(2,randSample[i]);        
    }
    
////////////////////////////////////////////////////////////
#else //EDGEVO_SUBSET_POINTS_EXACT
    // For non random numbers EDGEVO_SUBSET_POINTS
    size_t numElements = (m_finalMask.array() != 0).count();
    m_gxFinal.resize(numElements);
    m_gyFinal.resize(numElements);
    m_im1.resize(numElements);
    m_im2Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);

    size_t idx = 0;
    for(int i = 0; i < m_finalMask.rows(); ++i)
    {
        if(m_finalMask[i] != 0)
        {
            m_gxFinal[idx]  = interpolateVector( m_gx, m_warpedX[i], m_warpedY[i], w);
            m_gyFinal[idx]  = interpolateVector( m_gy, m_warpedX[i], m_warpedY[i], w);
            m_im1[idx] = m_im1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
            m_im2Final[idx] = interpolateVector(m_im2, m_warpedX[i], m_warpedY[i], w);
            m_XFinal[idx] = m_newX3D(0,i);
            m_YFinal[idx] = m_newX3D(1,i);
            m_ZFinal[idx] = m_newX3D(2,i);
            
            ++idx;
        }
    }
#endif //EDGEVO_SUBSET_POINTS_EXACT
////////////////////////////////////////////////////////////
    
    //apply mask to im1, im2, gx, and gy
    //interp coordinates of im2, gx, and gy
    // calc residual

    //calc A and b matrices
    //
    m_residual.resize(numElements);
    m_rsquared.resize(numElements);
    m_weights.resize(numElements);

    m_residual = ( m_im1.array() - m_im2Final.array() );
    m_rsquared = m_residual.array() * m_residual.array();

    m_weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
    m_weights = ( ( (m_residual.array()).abs() ) > EdgeVO::Settings::HUBER_THRESH ).select( EdgeVO::Settings::HUBER_THRESH / (m_residual.array()).abs() , m_weights);

    return ( (m_weights.array() * m_rsquared.array()).sum() / (float) numElements );
     
}

void EdgeDirectVO::solveSystemOfEquations(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdate)
{
    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);

    size_t numElements = m_im2Final.rows();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> Z2 = m_ZFinal.array() * m_ZFinal.array();

    m_Jacobian.resize(numElements, Eigen::NoChange);
    m_Jacobian.col(0) =  m_weights.array() * fx * ( m_gxFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(1) =  m_weights.array() * fy * ( m_gyFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(2) = - m_weights.array()* ( fx * ( m_XFinal.array() * m_gxFinal.array() ) + fy * ( m_YFinal.array() * m_gyFinal.array() ) )
                        / ( Z2.array() );

    m_Jacobian.col(3) = - m_weights.array() * ( fx * m_XFinal.array() * m_YFinal.array() * m_gxFinal.array() / Z2.array()
                         + fy *( 1.f + ( m_YFinal.array() * m_YFinal.array() / Z2.array() ) ) * m_gyFinal.array() );

    m_Jacobian.col(4) = m_weights.array() * ( fx * (1.f + ( m_XFinal.array() * m_XFinal.array() / Z2.array() ) ) * m_gxFinal.array() 
                        + fy * ( m_XFinal.array() * m_YFinal.array() * m_gyFinal.array() ) / Z2.array() );

    m_Jacobian.col(5) = m_weights.array() * ( -fx * ( m_YFinal.array() * m_gxFinal.array() ) + fy * ( m_XFinal.array() * m_gyFinal.array() ) )
                        / m_ZFinal.array();
    
    m_residual.array() *= m_weights.array();
    
    poseupdate = -( (m_Jacobian.transpose() * m_Jacobian).cast<double>() ).ldlt().solve( (m_Jacobian.transpose() * m_residual).cast<double>() );

    
}
float EdgeDirectVO::interpolateVector(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>& toInterp, float x, float y, int w) const
{
    int xi = (int) x;
	int yi = (int) y;
	float dx = x - xi;
	float dy = y - yi;
	float dxdy = dx * dy;
    int topLeft = w * yi + xi;
    int topRight = topLeft + 1;
    int bottomLeft = topLeft + w;
    int bottomRight= bottomLeft + 1;
  
    //               x                x+1
    //       ======================================
    //  y    |    topLeft      |    topRight      |
    //       ======================================
    //  y+w  |    bottomLeft   |    bottomRight   |
    //       ======================================
    return  dxdy * toInterp[bottomRight]
	        + (dy - dxdy) * toInterp[bottomLeft]
	        + (dx - dxdy) * toInterp[topRight]
			+ (1.f - dx - dy + dxdy) * toInterp[topLeft];
}
void EdgeDirectVO::prepare3DPoints( )
{
    
    for (int lvl = 0; lvl < EdgeVO::Settings::PYRAMID_DEPTH; ++lvl)
    {
        const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
        int w = m_sequence.getFrameWidth(lvl);
        int h = m_sequence.getFrameHeight(lvl);
        const float fx = cameraMatrix.at<float>(0, 0);
        const float cx = cameraMatrix.at<float>(0, 2);
        const float fy = cameraMatrix.at<float>(1, 1);
        const float cy = cameraMatrix.at<float>(1, 2);
        const float fxInv = 1.f / fx;
        const float fyInv = 1.f / fy;
    
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                int idx = y * w + x;
                m_X3DVector[lvl].row(idx) << (x - cx) * fxInv, (y - cy) * fyInv, 1.f ;
            }
        }
    }
}

void EdgeDirectVO::warpAndCalculateResiduals(const Pose& pose, const std::vector<float>& Z, const std::vector<bool>& E, const int h, const int w, const cv::Mat& cameraMatrix, const int lvl)
{
    const int ymax = h;
    const int xmax = w;
    const int length = xmax * ymax;

    const float fx = cameraMatrix.at<float>(0,0);
    const float cx = cameraMatrix.at<float>(0,2);
    const float fy = cameraMatrix.at<float>(1,1);
    const float cy = cameraMatrix.at<float>(1,2);

    const Mat inPose( m_trajectory.getCurrentPose().inversePose() );
    Eigen::Matrix<float,4,4> invPose;
    cv::cv2eigen(inPose,invPose);


    for(int i = 0; i < ymax*xmax; ++i)
    {
        float z3d = Z[i];
        float x = i / ymax;
        float y = i % xmax;
        float x3d = z3d * (x - cx)/ fx;
        float y3d = z3d * (y - cy)/ fy;
    }
    return;
}

inline
bool EdgeDirectVO::checkBounds(float x, float xlim, float y, float ylim, float oldZ, float newZ, bool edgePixel)
{
    return ( (edgePixel) & (x >= 0) & x < xlim & y >= 0 & y < ylim & oldZ >= 0. & newZ >= 0. );
        
}
void EdgeDirectVO::terminationRequested()
{
    printf("Display Terminated by User\n");
    m_statistics.printStatistics();

}

void EdgeDirectVO::outputPose(const Pose& pose, double timestamp)
{
    Eigen::Matrix<double,4,4,Eigen::RowMajor> T;
    cv::Mat pmat = pose.getPoseMatrix();
    cv::cv2eigen(pmat,T);
    Eigen::Matrix<double,3,3,Eigen::RowMajor> R = T.block<3,3>(0,0);
    Eigen::Matrix<double,3,Eigen::RowMajor> t = T.block<3,1>(0,3);
    Eigen::Quaternion<double> quat(R);

    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << timestamp;
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[0];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[1];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[2];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.x();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.y();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.z();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.w();
    m_outputFile << std::endl;
}



} //end namespace EdgeVO