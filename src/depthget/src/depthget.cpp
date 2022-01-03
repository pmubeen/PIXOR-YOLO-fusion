#include "ros/ros.h"
// darknet_ros_msgs
#include <depthget/BoundingBoxes.h>
#include <depthget/BoundingBox.h>
#include <depthget/BboxL.h>
#include <depthget/BboxLes.h>
#include <kitti_player/BboxL.h>
#include <kitti_player/BboxLes.h>
#include <iostream>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
//opencv
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <stdio.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <signal.h>

std::vector<float> precision;
std::vector<float> recall;
std::vector<float> average_iou;
std::vector<float> time_delay;
std::vector<int> frame_num;
std::vector<int> misclass;

void savetoFile(int sig)
{
        //saving results to csv file
        printf("exporting results to csv \n");
        std::ofstream results;
        results.open("results.csv");
        char headers[] = "frame,precision,recall,avg_iou,time_delay,misclassifications \n";
        results << headers;
        for (int i = 0; i < frame_num.size(); i++)
        {
                results << frame_num[i] << "," << precision[i] << "," << recall[i] << ","
                        << average_iou[i] << "," << time_delay[i] << "," << misclass[i] << "\n";
        }
        results.close();
        printf("done!\n");
        ros::shutdown();
}

std::string str_tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), 
                   [](unsigned char c){ return std::tolower(c); } // correct
                  );
    return s;
}

class DepthHandler
{
public:
        DepthHandler() : it(nh)
        {
                truth_sub = nh.subscribe("/kitti_player/ground_truth", 10, &DepthHandler::Truthcb, this);
                sub1 = it.subscribe("/kitti_player/color/left/image_rect", 200, &DepthHandler::PictureCB, this);
                sub2 = nh.subscribe("/darknet_ros/bounding_boxes", 200, &DepthHandler::BBcb, this);
                sub3 = nh.subscribe("/pixor_node/bbox", 10, &DepthHandler::ROIL2Dcb, this);

                pub1 = it.advertise("depthMap", 1);
        }
        std::vector<depthget::BboxL> BBL;
        cv::Mat dstImage;
        cv_bridge::CvImagePtr cv_ptr;
        std::vector<depthget::BoundingBox> BBoxes;
        std::vector<depthget::BoundingBox> Truth_BBoxes;
        //velo to cam projection matrix
        float proj_mat[3][4] = {
            {609.6954, -721.4216, -1.2513, -123.0418},
            {180.3842, 7.6448, -719.6515, -101.0167},
            {0.9999, 0.0001, 0.0105, -0.2694}};

        void PictureCB(const sensor_msgs::ImageConstPtr &msg)
        {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                dstImage = cv_ptr->image;

                //drawing pixor bounding boxes
                for (int i = 0; i < BBL.size(); ++i)
                {
                        float length = BBL[i].maxx - BBL[i].minx;
                        float width = BBL[i].maxy - BBL[i].miny;

                        cv::rectangle(dstImage, cv::Rect(BBL[i].minx, BBL[i].miny, length, width), cv::Scalar(255, 0, 0), 1, 1, 0);
                }

                //publishing yolo bounding boxes
                for (int i = 0; i < BBoxes.size(); ++i)
                {
                        if (BBoxes[i].probability > 0.4)
                        {

                                float length = BBoxes[i].xmax - BBoxes[i].xmin;
                                float width = BBoxes[i].ymax - BBoxes[i].ymin;
                                cv::rectangle(dstImage, cv::Rect(BBoxes[i].xmin, BBoxes[i].ymin, length, width), cv::Scalar(0, 0, 255), 1, 1, 0);
                        }
                }

                float act_detect = 0; //actual number of objects in ground truth

                //drawing ground truth bounding boxes
                for (int i = 0; i < Truth_BBoxes.size(); i++)
                {
                        act_detect++;
                        int length = Truth_BBoxes[i].xmax - Truth_BBoxes[i].xmin;
                        int width = Truth_BBoxes[i].ymax - Truth_BBoxes[i].ymin;
                        cv::rectangle(dstImage, cv::Rect(Truth_BBoxes[i].xmin, Truth_BBoxes[i].ymin, length, width), cv::Scalar(0, 255, 0), 1, 1, 0);
                }

                //Fusion of pixor and yolo bounding boxes
                finalbox fusionfbox;
                float true_pos = 0;
                float sys_detect = 0;
                float iou;
                int misclassifications = 0;

                for (int i = 0; i < BBoxes.size(); ++i)
                {
                        //filtering pixor bounding boxes
                        if (BBoxes[i].probability > 0.4)
                        {
                                float lengthc = BBoxes[i].xmax - BBoxes[i].xmin;
                                float widthc = BBoxes[i].ymax - BBoxes[i].ymin;

                                iou = 0;

                                for (int k = 0; k < BBL.size(); ++k)
                                {
                                        float lengthl = BBL[k].maxx - BBL[k].minx;
                                        float widthl = BBL[k].maxy - BBL[k].miny;

                                        bool fumaxx = (BBoxes[i].xmax - BBL[k].maxx < lengthc / 2) && (BBoxes[i].xmax - BBL[k].maxx > -lengthc / 2);
                                        bool fuminx = (BBoxes[i].xmin - BBL[k].minx < lengthc / 2) && (BBoxes[i].xmin - BBL[k].minx > -lengthc / 2);
                                        bool fuminy = (BBoxes[i].ymin - BBL[k].miny < widthc / 2) && (BBoxes[i].ymin - BBL[k].miny > -widthc / 2);
                                        bool fumaxy = (BBoxes[i].ymax - BBL[k].maxy < widthc / 2) && (BBoxes[i].ymax - BBL[k].maxy > -widthc / 2);

                                        if (fumaxx && fuminx && fuminy && fumaxy)
                                        {
                                                sys_detect++; //append to num of objects detected by fusion system
                                                fusionfbox.maxx = (BBoxes[i].xmax + BBL[k].maxx) / 2;
                                                fusionfbox.maxy = (BBoxes[i].ymax + BBL[k].maxy) / 2;
                                                fusionfbox.minx = (BBoxes[i].xmin + BBL[k].minx) / 2;
                                                fusionfbox.miny = (BBoxes[i].ymin + BBL[k].miny) / 2;
                                                fusionfbox.centerx = BBL[k].centerx;
                                                fusionfbox.centery = BBL[k].centery;
                                                fusionfbox.navi = BBL[k].navi;

                                                fusionfbox.Class = BBoxes[i].Class;
                                                float lengthf = fusionfbox.maxx - fusionfbox.minx;
                                                float widthf = fusionfbox.maxy - fusionfbox.miny;
                                                cv::rectangle(dstImage, cv::Rect(fusionfbox.minx, fusionfbox.miny, lengthf, widthf), cv::Scalar(0, 0, 255), 3, 1, 0);

                                                //testing against ground truth
                                                for (int j = 0; j < Truth_BBoxes.size(); j++)
                                                {
                                                        //finding IoU
                                                        float xinter[2] = {0, 0};
                                                        float yinter[2] = {0, 0};
                                                        if (Truth_BBoxes[j].xmin > fusionfbox.minx)
                                                        {
                                                                xinter[0] = Truth_BBoxes[j].xmin;
                                                                xinter[1] = fusionfbox.maxx;
                                                        }
                                                        if (fusionfbox.minx > Truth_BBoxes[j].xmin)
                                                        {
                                                                xinter[0] = fusionfbox.minx;
                                                                xinter[1] = Truth_BBoxes[j].xmax;
                                                        }
                                                        if (Truth_BBoxes[j].ymin > fusionfbox.miny)
                                                        {
                                                                yinter[0] = Truth_BBoxes[j].ymin;
                                                                yinter[1] = fusionfbox.maxy;
                                                        }
                                                        if (fusionfbox.miny > Truth_BBoxes[j].ymin)
                                                        {
                                                                yinter[0] = fusionfbox.miny;
                                                                yinter[1] = Truth_BBoxes[j].ymax;
                                                        }
                                                        //calculating intersection and union area
                                                        float interarea = (xinter[1] - xinter[0]) * (yinter[1] - yinter[0]);
                                                        float unionarea = (Truth_BBoxes[j].xmax - Truth_BBoxes[j].xmin) * (Truth_BBoxes[j].ymax - Truth_BBoxes[j].ymin) +
                                                                          (fusionfbox.maxx - fusionfbox.minx) * (fusionfbox.maxy - fusionfbox.miny);
                                                        unionarea -= interarea;

                                                        //IOU threshold
                                                        if (interarea / unionarea > 0.7)
                                                        {
                                                                iou += interarea / unionarea; //append iou measurement
                                                                true_pos++;                   //append to the number of the true positives
                                                                //- detected by fusion system and passing the IOU threshold
                                                                cv::rectangle(dstImage, cv::Rect(fusionfbox.minx, fusionfbox.miny, lengthf, widthf), cv::Scalar(0, 255, 255), 2, 0, 0);
                                                                
                                                                if (str_tolower(Truth_BBoxes[j].Class) != fusionfbox.Class)
                                                                {
                                                                        //if there are misclassifications, add to count
                                                                        misclassifications++;
                                                                }
                                                        }
                                                }

                                                std::string text = fusionfbox.Class + "-" + "navi:" + std::to_string(int(fusionfbox.navi)) + " " + "x:" + std::to_string(int(fusionfbox.centerx)) + " " + "y:" + std::to_string(int(fusionfbox.centery));

                                                int font_face = cv::FONT_HERSHEY_COMPLEX;
                                                double font_scale = 0.5;
                                                int thickness = 1;
                                                int baseline;

                                                cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
                                                cv::Point origin;
                                                origin.x = fusionfbox.minx;
                                                origin.y = fusionfbox.miny - text_size.height / 2;
                                                cv::putText(dstImage, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

                                                break;
                                        }
                                }
                        }
                }
                sensor_msgs::ImagePtr pubmsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", dstImage).toImageMsg();
                float false_pos = sys_detect - true_pos; //calculate false positives
                float false_neg = act_detect - true_pos; // calculate true negatives

                if (true_pos + false_pos == 0)
                {
                        precision.push_back(NULL); //no detections by the system
                }
                else
                {
                        precision.push_back(true_pos / (true_pos + false_pos)); //calculate precision for the frame
                }

                if (true_pos + false_neg == 0)
                {
                        recall.push_back(NULL); //no detections and no actual objects to detect
                }
                else
                {
                        recall.push_back(true_pos / (true_pos + false_neg)); //calculate recall for the frame
                }

                average_iou.push_back(iou / true_pos); //calculate average iou for the frame
                time_delay.push_back(msg->header.stamp.toSec() - ros::Time::now().toSec());
                frame_num.push_back(msg->header.seq);
                misclass.push_back(misclassifications);

                Truth_BBoxes.clear();
                pub1.publish(pubmsg);
                dstImage.release();
                BBoxes.clear();
        }

        void ROIL2Dcb(const depthget::BboxLes::ConstPtr &input)
        {
                BBL = input->bboxl;
        }
        void BBcb(const depthget::BoundingBoxes::ConstPtr &msg)
        {
                BBoxes = msg->bounding_boxes;
        }

        void matmul(float mat1[3][4], float mat2[4][8], float rslt[3][8])
        {

                for (int i = 0; i < 3; i++)
                {
                        for (int j = 0; j < 8; j++)
                        {
                                rslt[i][j] = 0;

                                for (int k = 0; k < 4; k++)
                                {
                                        rslt[i][j] += mat1[i][k] * mat2[k][j];
                                }
                        }
                }
        }

        void velo2cam(float points[4][8], float &xmax, float &xmin, float &ymax, float &ymin) //projecting velodyne points into camera frame
        {
                float cam[3][8];

                matmul(proj_mat, points, cam);

                std::vector<float> x, y; //projected velodyne points
                for (int i = 0; i < 8; i++)
                {
                        cam[0][i] = cam[0][i] / cam[2][i]; // x / z
                        cam[1][i] = cam[1][i] / cam[2][i]; // y / z
                        x.push_back(cam[0][i]);
                        y.push_back(cam[1][i]);
                }

                xmax = *std::max_element(x.begin(), x.end());
                xmin = *std::min_element(x.begin(), x.end());
                ymax = *std::max_element(y.begin(), y.end());
                ymin = *std::min_element(y.begin(), y.end());
        }

        void Truthcb(const kitti_player::BboxLes::ConstPtr &input)
        {
                float minx, miny, maxx, maxy, minz, maxz;
                float xmax, xmin, ymax, ymin;
                depthget::BoundingBox Truth_BBox;
                for (int i; i < input->bboxl.size(); i++)
                {
                        minx = input->bboxl[i].minx;
                        maxx = input->bboxl[i].maxx;
                        miny = input->bboxl[i].miny;
                        maxy = input->bboxl[i].maxy;
                        minz = input->bboxl[i].minz;
                        maxz = input->bboxl[i].maxz;

                        float points[4][8] = {{minx, minx, maxx, maxx, minx, minx, maxx, maxx},
                                              {miny, maxy, miny, maxy, miny, maxy, miny, maxy},
                                              {minz + 1, minz + 1, minz + 1, minz + 1, maxz + 1, maxz + 1, maxz + 1, maxz + 1}, //accounting for height offset in cam
                                              {1, 1, 1, 1, 1, 1, 1, 1}};

                        velo2cam(points, xmax, xmin, ymax, ymin);

                        //confining bounding box within camera frame
                        if (xmax > 1241)
                        {
                                xmax = 1241;
                        }
                        else if (xmax < 0)
                        {
                                xmax = 0;
                        }

                        if (xmin > 1241)
                        {
                                xmin = 1241;
                        }
                        else if (xmin < 0)
                        {
                                xmin = 0;
                        }

                        if (ymax > 374)
                        {
                                ymax = 374;
                        }
                        else if (ymax < 0)
                        {
                                ymax = 0;
                        }

                        if (ymin > 374)
                        {
                                ymin = 374;
                        }
                        else if (ymin < 0)
                        {
                                ymin = 0;
                        }
                        Truth_BBox.xmax = xmax;
                        Truth_BBox.xmin = xmin;
                        Truth_BBox.ymax = ymax;
                        Truth_BBox.ymin = ymin;
                        Truth_BBox.Class = input->bboxl[i].Class;
                        Truth_BBoxes.push_back(Truth_BBox);
                }
        }

protected:
        ros::NodeHandle nh;
        ros::Subscriber sub2, sub3, truth_sub;
        image_transport::ImageTransport it;
        image_transport::Subscriber sub1;
        image_transport::Publisher pub1;
        struct Bbox
        {
                float minx_bb = 0;
                float maxx_bb = 0;
                float miny_bb = 0;
                float maxy_bb = 0;
                int flag_bb = 0;
        };
        struct finalbox
        {
                float maxx = 0;
                float maxy = 0;
                float minx = 0;
                float miny = 0;
                float centerx = 0;
                float centery = 0;
                float navi = 0;
                std::string Class;
        };
};

int main(int argc, char **argv)
{
        // Initialize ROS
        ros::init(argc, argv, "depthget", ros::init_options::NoSigintHandler);

        DepthHandler handler;

        signal(SIGINT, savetoFile);

        // Spin
        ros::spin();
        return 0;
}
