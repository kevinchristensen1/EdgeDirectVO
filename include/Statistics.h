#ifndef STATISTICS_H
#define STATISTICS_H
#include <opencv2/core/core.hpp>
#include <vector>
#include "Settings.h"
#include "CycleTimer.h"


namespace EdgeVO{
class Statistics{
    public:
        Statistics();
        Statistics(const Statistics& cp);
        ~Statistics();

        //Statistics functions
        float getAverageLatency() const;
        float getMedianLatency(); //calls nth_element
        float getStdDevLatency() const;
        float getMinLatency() const;
        float getMaxLatency() const;

        //Updating functions
        void start();
        void end();
        void addDurationForFrame(float startTime, float endTime);
        void addStartTime(float startTime);
        void addCurrentTime(float currTime);
        
        //Printing functions
        void printStatistics() const;
        void printFinalStatistics();
        void printEvalStatement() const;

        void incrementIterationCount() {++m_numberImages;}
        

    private:
        //Statistics& operator=(const Statistics& rhs) = 0;
        float m_runtimePerLevel[EdgeVO::Settings::PYRAMID_DEPTH];
        std::vector<float> m_frameDurations;
        float durationMean;
        
        float m_startTime;
        float m_endTime;
        int m_numberImages;

};
// Inline functions
inline void Statistics::start()
{
    m_startTime = EdgeVO::CycleTimer::currentSeconds();
}

inline void Statistics::end()
{
    m_endTime = EdgeVO::CycleTimer::currentSeconds();
}

}

#endif //STATISTICS_H