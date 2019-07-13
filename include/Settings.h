#ifndef SETTINGS_H
#define SETTINGS_H

#include <iostream>
#include <math.h>
#include <string>

namespace EdgeVO{
    namespace Settings{
        // Macros are used so we do not incur any speed costs (ie. avoid class polymorphism v tables)
        // and control everything from here, rather than in main().
        
        //#define DISPLAY_SEQUENCE true         //For displaying, uncomment this

        #define CANNY_EDGES true
        //#define LoG_EDGES true
        //#define SOBEL_EDGES true
        //#define SFORESTS_EDGES true

        //#define CONV_BASIN true

        //#define REGULAR_DIRECT_VO true
        //#define REGULAR_DIRECT_VO_SUBSET true
        //#define EDGEVO_SUBSET_POINTS true
        const double PERCENT_EDGES = 0.85;

        //const int NUMBER_POINTS = 50;
        
        //#define EDGEVO_SUBSET_POINTS_EXACT true     //For only using a subset of points, uncomment this

        // Sequence.h //
        // Specify datasets
        enum SELECT_DATASET
        {
            rgbd_dataset_freiburg1_xyz =    0,
            rgbd_dataset_freiburg2_xyz =    1,
            rgbd_dataset_freiburg1_desk =   2,
            rgbd_dataset_freiburg2_desk =   3,
            rgbd_dataset_freiburg3_long_office_household = 4,
            rgbd_dataset_freiburg1_desk2 = 5,
            rgbd_dataset_freiburg2_large_no_loop = 6,
            rgbd_dataset_freiburg3_structure_notexture_far = 7,
            rgbd_dataset_freiburg3_structure_notexture_near = 8,
            rgbd_dataset_freiburg1_rpy = 9,
            rgbd_dataset_freiburg1_room = 10,
            rgbd_dataset_freiburg1_plant = 11
        };
        const std::string DATASET_OPTIONS[] = 
        {   std::string("rgbd_dataset_freiburg1_xyz"), 
            std::string("rgbd_dataset_freiburg2_xyz"),
            std::string("rgbd_dataset_freiburg1_desk"), 
            std::string("rgbd_dataset_freiburg2_desk"),
            std::string("rgbd_dataset_freiburg3_long_office_household"),
            std::string("rgbd_dataset_freiburg1_desk2"),
            std::string("rgbd_dataset_freiburg2_large_no_loop"),
            std::string("rgbd_dataset_freiburg3_structure_notexture_far"),
            std::string("rgbd_dataset_freiburg3_structure_notexture_near"), 
            std::string("rgbd_dataset_freiburg1_rpy"),
            std::string("rgbd_dataset_freiburg1_room"),
            std::string("rgbd_dataset_freiburg1_plant")
        };

        const int DATASET_NUMBER = 1;

        const int KEYFRAME_INTERVAL(3);

        const int PYRAMID_DEPTH(3);


        const std::string DATASET(DATASET_OPTIONS[DATASET_NUMBER]);
        // const std::string DATASET("rgbd_dataset_freiburg1_xyz");

        //Image and Depth Map Directories
        const std::string DATASET_DIRECTORY( std::string("../") + DATASET + std::string("/") );
        const std::string ASSOC_FILE( DATASET_DIRECTORY + std::string("assoc.txt") );

        const std::string IMAGE_DIRECTORY(DATASET_DIRECTORY + std::string("rgb") );
        const std::string DEPTH_DIRECTORY(DATASET_DIRECTORY + std::string("depth") );

        // OpenCV Window name and termination key.  Used by Sequence::displaySequence()
        const std::string DISPLAY_SEQUENCE_WINDOW("Display Sequence");
        const std::string DISPLAY_EDGE_WINDOW("Edges");
        const std::string DISPLAY_DEPTH_WINDOW("Depth");

        const int TERMINATE_DISPLAY_KEY(27); // 27 = Esc key
        const int DISPLAY_WINDOW_CLOSED(-1); //-1 = Window is closed

        
        const int PYRAMID_OUT_OF_BOUNDS(PYRAMID_DEPTH - 1);
        const int PYRAMID_BUILD(PYRAMID_DEPTH - 1);
        const int MAX_ITERATIONS_PER_PYRAMID[] = {10, 15, 20}; //{10,15,20}


        const float DEPTH_SCALE_FACTOR(5000.); //For TUM RGBD this is 5000.
        const int SOBEL_KERNEL_SIZE(1);

        const double MIN_TRANSLATION_UPDATE(1.e-8);
        const double MIN_ROTATION_UPDATE(1.e-8);
        const double EPSILON(1.e-8);

        const float MAX_Z_DEPTH(10.f);

        // const float FX = 517.3f;
        // const float FY = 516.5f;
        // const float CX = 318.6f;
        // const float CY = 255.3f;

        // const float FX = 525.f;
        // const float FY = 525.f;
        // const float CX = 319.5f;
        // const float CY = 239.5f;

        const float FX = 520.9f;
        const float FY = 521.0f;
        const float CX = 325.1f;
        const float CY = 249.7f;

        const float CANNY_RATIO(0.5);//0.33 //0.5 //matlab uses 0.4 //0.15 working well
        const double SIGMA(sqrt(2.));
        //Statistics Class
        const float SECONDS_TO_MILLISECONDS(1000.f);
        const double PIXEL_TO_METER_SCALE_FACTOR(0.0002); // 0.0002f = 1.f/5000.f

        //Trajectory Class
        const std::string GROUNDTRUTH_FILE(DATASET_DIRECTORY + std::string("groundtruth.txt"));


        //General
        const float INF_F(std::numeric_limits<float>::infinity());

        //EdgeVO.h Huber Weights
        const float HUBER_THRESH = 5.f;
        // EdgeVO.h Lambda
        const float LAMBDA_MIN = 0.f;
        const float LAMBDA_MAX = 0.2f;
        const float LAMBDA_UPDATE_FACTOR = 0.5f;

        const float MIN_GRADIENT_THRESH = 50.f;
        //EdgeVO.h output file
        const std::string RESULTS_FILE(std::string("../") + DATASET + std::string("_results.txt") );
        const int RESULTS_FILE_PRECISION(7);

        //Number of points to use in EdgeVO.cpp
        



        const std::string DATASET_EVAL_RPE_OPTIONS[] = 
        {   std::string("python evaluate_rpe.py groundtruth_fr1xyz.txt rgbd_dataset_freiburg1_xyz_results.txt --fixed_delta --delta 1 --verbose") , 
            std::string("python evaluate_rpe.py groundtruth_fr2xyz.txt rgbd_dataset_freiburg2_xyz_results.txt --fixed_delta --delta 1 --verbose") ,
            std::string("python evaluate_rpe.py groundtruth_fr1desk.txt rgbd_dataset_freiburg1_desk_results.txt --fixed_delta --delta 1 --verbose") , 
            std::string("python evaluate_rpe.py groundtruth_fr2desk.txt rgbd_dataset_freiburg2_desk_results.txt --fixed_delta --delta 1 --verbose") ,
            std::string("python evaluate_rpe.py groundtruth_fr3long.txt rgbd_dataset_freiburg3_long_office_household_results.txt --fixed_delta --delta 1 --verbose") ,
            std::string("python evaluate_rpe.py groundtruth_fr1desk2.txt rgbd_dataset_freiburg1_desk2_results.txt --fixed_delta --delta 1 --verbose") ,
            std::string("python evaluate_rpe.py groundtruth_fr2large.txt rgbd_dataset_freiburg2_large_no_loop_results.txt --fixed_delta --delta 1 --verbose")  ,
            std::string("python evaluate_rpe.py groundtruth_fr3stru.txt rgbd_dataset_freiburg3_structure_notexture_far_results.txt --fixed_delta --delta 1 --verbose"),
            std::string("python evaluate_rpe.py groundtruth_fr3stru_near.txt rgbd_dataset_freiburg3_structure_notexture_near_results.txt --fixed_delta --delta 1 --verbose"),
            std::string("python evaluate_rpe.py groundtruth_fr1rpy.txt rgbd_dataset_freiburg1_rpy_results.txt --fixed_delta --delta 1 --verbose") ,
            std::string("python evaluate_rpe.py groundtruth_fr1room.txt rgbd_dataset_freiburg1_room_results.txt --fixed_delta --delta 1 --verbose"),
            std::string("python evaluate_rpe.py groundtruth_fr1plant.txt rgbd_dataset_freiburg1_plant_results.txt --fixed_delta --delta 1 --verbose")
            };
        const std::string DATASET_EVAL_RPE( DATASET_EVAL_RPE_OPTIONS[DATASET_NUMBER]);

        const std::string DATASET_EVAL_ATE_OPTIONS[] = 
        {   std::string("python evaluate_ate.py groundtruth_fr1xyz.txt rgbd_dataset_freiburg1_xyz_results.txt --verbose") , 
            std::string("python evaluate_ate.py groundtruth_fr2xyz.txt rgbd_dataset_freiburg2_xyz_results.txt --verbose") ,
            std::string("python evaluate_ate.py groundtruth_fr1desk.txt rgbd_dataset_freiburg1_desk_results.txt --verbose") , 
            std::string("python evaluate_ate.py groundtruth_fr2desk.txt rgbd_dataset_freiburg2_desk_results.txt --verbose") ,
            std::string("python evaluate_ate.py groundtruth_fr3long.txt rgbd_dataset_freiburg3_long_office_household_results.txt --verbose") ,
            std::string("python evaluate_ate.py groundtruth_fr1desk2.txt rgbd_dataset_freiburg1_desk2_results.txt --verbose") ,
            std::string("python evaluate_ate.py groundtruth_fr2large.txt rgbd_dataset_freiburg2_large_no_loop_results.txt --verbose")  ,
            std::string("python evaluate_ate.py groundtruth_fr3stru.txt rgbd_dataset_freiburg3_structure_notexture_far_results.txt --verbose"),
            std::string("python evaluate_ate.py groundtruth_fr3stru_near.txt rgbd_dataset_freiburg3_structure_notexture_near_results.txt --verbose"),
            std::string("python evaluate_ate.py groundtruth_fr1rpy.txt rgbd_dataset_freiburg1_rpy_results.txt --verbose") ,
            std::string("python evaluate_ate.py groundtruth_fr1room.txt rgbd_dataset_freiburg1_room_results.txt --verbose"),
            std::string("python evaluate_ate.py groundtruth_fr1plant.txt rgbd_dataset_freiburg1_plant_results.txt --verbose")
            };
        const std::string DATASET_EVAL_ATE( DATASET_EVAL_ATE_OPTIONS[DATASET_NUMBER]);

    }
}

#endif //SETTINGS_H
