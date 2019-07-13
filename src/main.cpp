#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Eigen/Dense"
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <utility>
#include <functional>
#include <cctype>
#include <cstdlib>
#include <string>

#include "EdgeDirectVO.h"
//#include "EdgeDirectVOMultiThread.h"
#include "CycleTimer.h"
#include "Sequence.h"
#include "Settings.h"
#include "Pose.h"
#include "Trajectory.h"


#define Eigen_Vectorize
/*
std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}
std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}
std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}

int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != "..")
    		files.push_back(name);
    }
    closedir(dp);


    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(unsigned int i=0;i<files.size();i++)
	{
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}

int getFile (std::string source, std::vector<std::string> &files)
{
	std::ifstream f(source.c_str());

	if(f.good() && f.is_open())
	{
		while(!f.eof())
		{
			std::string l;
			std::getline(f,l);

			l = trim(l);

			if(l == "" || l[0] == '#')
				continue;

			files.push_back(l);
		}

		f.close();

		size_t sp = source.find_last_of('/');
		std::string prefix;
		if(sp == std::string::npos)
			prefix = "";
		else
			prefix = source.substr(0,sp);

		for(unsigned int i=0;i<files.size();i++)
		{
			if(files[i].at(0) != '/')
				files[i] = prefix + "/" + files[i];
		}

		return (int)files.size();
	}
	else
	{
		f.close();
		return -1;
	}

}
*/
using namespace cv;
using namespace std;
using namespace Eigen;

int main()
{
	/*
    Mat img = imread("../rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png");
	std::string source = "../rgbd_dataset_freiburg1_xyz/rgb";
	std::vector<std::string> files;
	*/

	/*
	if(getdir(source, files) >= 0)
	{
		printf("found %d image files in folder %s!\n", (int)files.size(), source.c_str());
	}
	else if(getFile(source, files) >= 0)
	{
		printf("found %d image files in file %s!\n", (int)files.size(), source.c_str());
	}
	else
	{
		printf("could not load file list! wrong path / file?\n");
	}

	*/

	//EdgeVO::Sequence seq;
	//seq.displaySequence();

	/*
	EdgeVO::Sequence seq;

	for(int i = 0; i <10; i++)
	{
		double s = EdgeVO::CycleTimer::currentSeconds();
		//Mat a(seq.getReferenceFrame().getImage(0));
		double denominator(std::pow(2., i));
		double e = EdgeVO::CycleTimer::currentSeconds();
		std::cout << denominator << std::endl;
		printf("[Vector Timing]:\t\t[%.3f] ms\n", (e - s) * 1000);
	}

	EdgeVO::Pose pose1;
	EdgeVO::Pose pose2;
	pose2 = pose2 / 2.;
	std::cout << pose1.getPoseMatrix() << std::endl;
	std::cout << pose2.getPoseMatrix() << std::endl;
	std::cout << (pose1*pose2).getPoseMatrix() << std::endl;

	double data[6] = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
	cv::Mat input = cv::Mat(6, 1, CV_64FC1, data);

	cv::Mat toReturn;
	EdgeVO::se3Exp(input, toReturn);
	std::cout << toReturn << std::endl;

	*/
	
	EdgeVO::EdgeDirectVO evo;
	//evo.runEdgeDirectVO();

	//EdgeVO::EdgeDirectVOMultiThread evo;
	evo.runEdgeDirectVO();

	//EdgeVO::Trajectory t;
	
	/*
	double data[16] = {0.0754  ,  0.6139 , -0.7857, 1.3405, 0.9971  , -0.0384  ,  0.0657 , 0.6266, 0.0102 ,  -0.7884 ,  -0.6150,  1.6575, 0.,0.,0.,1.};
	cv::Mat a(4,4, CV_64FC1, data);
	EdgeVO::Pose p1(a);
	cv::Mat b(4,4, CV_64FC1);
	p1.inversePose(b);
	std::cout << p1 << std::endl;
	std::cout << b << std::endl;
	*/

	// EdgeVO::Pose p1;
	// EdgeVO::Pose p2;
	// p1-=2.;
	// p2/=2.;
	// std::cout << p1 << std::endl;
	// std::cout << p2 << std::endl;


	//h.displaySequence();

	//h.displaySequence();
    /*
	for(unsigned int i=0;i<files.size();i++)
	{
		cv::Mat imageDist = cv::imread(files[i], cv::ImreadModes::IMREAD_GRAYSCALE);
		double startTime = EdgeVO::CycleTimer::currentSeconds();

		cv::imshow( "Display window", imageDist );  
		cv::waitKey(1);
		double endTime = EdgeVO::CycleTimer::currentSeconds();
	
		printf("[Display Images]:\t\t[%.3f] ms\n", (endTime - startTime) * 1000);

	}
	*/
/*
      
	Matrix3f A;
	Vector3f b;
	A << 1.2,3.4,5.6,  7.8,9.0,1.2,  3.4,5.6,7.8;
	b << 36.4, 87.6, 62.8;
	cout << "Here is the matrix A:\n" << A << endl;
	cout << "Here is the vector b:\n" << b << endl;

	double startTime = EdgeVO::CycleTimer::currentSeconds();
	Vector3f x = A.colPivHouseholderQr().solve(b);
	cout << "The solution is:\n" << x << endl;        
	double endTime = EdgeVO::CycleTimer::currentSeconds();
	printf("[Vector Timing]:\t\t[%.3f] ms\n", (endTime - startTime) * 1000);


	EdgeVO::EdgeDirectVO evo;

	*/

    return 0;
}
