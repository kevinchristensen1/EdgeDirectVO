#include "Statistics.h"
#include <iostream>
#include "Settings.h"
#include "CycleTimer.h"
#include <vector>
#include <algorithm>
#include <numeric> //accumulate


namespace EdgeVO{

Statistics::Statistics()
    :m_runtimePerLevel{0.}, m_numberImages(0) , 
     m_startTime(0.) , m_endTime(0.)
{}

Statistics::~Statistics()
{
    printFinalStatistics();
}

float Statistics::getAverageLatency() const
{
    return std::accumulate(m_frameDurations.begin(), m_frameDurations.end(), 0) / (float) m_frameDurations.size();
}

float Statistics::getMedianLatency()
{
    std::nth_element (m_frameDurations.begin(), m_frameDurations.begin() + (m_frameDurations.size() / 2), m_frameDurations.end());
    return *(m_frameDurations.begin() + (m_frameDurations.size() / 2) );
}
float Statistics::getMinLatency() const
{
    return *std::min_element(m_frameDurations.begin(), m_frameDurations.end() );
}
float Statistics::getMaxLatency() const
{
    return *std::max_element(m_frameDurations.begin(), m_frameDurations.end() );
}


float Statistics::getStdDevLatency() const
{
    float sum = 0, mean = getAverageLatency();

    for(int i = 0; i < m_frameDurations.size(); ++i)
    {
        sum += std::pow( m_frameDurations[i] - mean, 2.);
    }
    return std::sqrt( sum / (float)  m_frameDurations.size() );
}


void Statistics::addDurationForFrame(float startTime, float endTime)
{

    float durationInMilliseconds = (endTime - startTime) * EdgeVO::Settings::SECONDS_TO_MILLISECONDS;
    m_frameDurations.push_back(durationInMilliseconds);
    incrementIterationCount();
}

void Statistics::addStartTime(float startTime)
{
    m_startTime = startTime;
}

void Statistics::addCurrentTime(float currTime)
{
    m_endTime = currTime;
}

void Statistics::printEvalStatement() const
{
    std::cout << "Run for evaluation: " << EdgeVO::Settings::DATASET_EVAL_RPE << std::endl;
    std::cout << EdgeVO::Settings::DATASET_EVAL_ATE << std::endl;
}
        

void Statistics::printStatistics() const
{
    printf("[Statistics]\n");
    //printf("[Vector Timing]:\t\t[%.3f] ms\n", (endTime - startTime) * 1000);
    printf("[Dataset]:\t\t\t[%s]\n", EdgeVO::Settings::DATASET.c_str());
    printf("[Image Number]:\t\t\t[%d images]\n", m_numberImages);
    printf("[Main Algorithm Loop]:\t\t[%.4f ms]\t\t[Current runtime]:\t\t[%.4f s]\n", m_frameDurations.back(), m_endTime - m_startTime);
    printf("[End Statistics]\n");
}


void Statistics::printFinalStatistics()
{
    printf("[Statistics]\n");
    printf("[Number Images]:\t\t[%d iterations]\n", m_numberImages);
    printf("[Main Algorithm Loop]:\t\t[%.4f ms]\n", m_frameDurations.back());
    printf("[Total Elapsed Time]:\t\t[%.4f s]\n", (m_endTime - m_startTime));
    printf("[Average Latency]:\t\t[%.4f ms]\n", getAverageLatency() );
    printf("[Median Latency]:\t\t[%.4f ms]\n", getMedianLatency() );
    printf("[Max Latency]:\t\t\t[%.4f ms]\n", getMaxLatency() );
    printf("[Min Latency]:\t\t\t[%.4f ms]\n", getMinLatency() );
    printf("[Std. Dev. Latency]:\t\t[%.4f ms]\n", getStdDevLatency() );
    printf("[End Statistics]\n");
    printEvalStatement();
    
}

}