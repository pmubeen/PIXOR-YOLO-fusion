#include <iostream>
#include <chrono>
#include <ctime>
#include <math.h>
#include <signal.h>
//ROS
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include "./preprocess.c"

class cloudHandler
{
public:
  cloudHandler()
  {
    pub1 = nh.advertise<sensor_msgs::PointCloud2>("ROI", 1000);
    pub2 = nh.advertise<sensor_msgs::PointCloud2>("ROIpoint", 1000);

    // Create a ROS subscriber for the input point cloud
    sub = nh.subscribe("kitti_player/hdl64e", 1000, &cloudHandler::cloud_cb, this);
  }

  float max(float a, float b)
  {
    float c;
    if (a > b)
      c = a;
    else
      c = b;
    return c;
  }
  float min(float a, float b)
  {
    float c;
    if (a < b)
      c = a;
    else
      c = b;
    return c;
  }

  void roi_bbox_gen(const void *input_arr, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
  {
    pcl::PointXYZI point;

    for (x = winsize / 2; x <= 700; x += winsize + 1)
    {
      for (y = winsize / 2; y <= 800; y += winsize + 1)
      {
        count = 0;

        for (i = -winsize / 2; i <= winsize / 2; i++)
        {
          for (j = -winsize / 2; j <= winsize / 2; j++)
          {
            if ((x + i) <= 700 && y + j <= 800)
            {
              inten = *(((float *)input_arr) + at3(y + j, x + i, 35));
              if (inten > (float)0)
              {
                count += 1;
              }
            }
          }
        }
        if (
            count >= 20
            // Course ground removal enabled then: (count >= 10 && count <= 15) || (count >= 30 && count <= 80)
        )
        {
          point.x = x - winsize / 2;
          point.y = y - winsize / 2;
          point.z = x + winsize / 2;
          point.intensity = y + winsize / 2;
          cloud->push_back(point);
        }
      }
    }
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = "base_link";
    cloud->header = pcl_conversions::toPCL(cloud_msg.header);
  }

  void pt_cloud_pub(const void *input_arr, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
  {
    pcl::PointXYZI point;
    count = 0;
    for (x = 0; x <= 700; x++)
    {
      for (y = 0; y <= 800; y++)
      {
        for (z = 0; z <= 34; z++)
        {
          point.x = (float)x;
          point.y = (float)y;
          point.z = z;
          point.intensity = *(((float *)input_arr) + at3(y, x, z));
          if (point.intensity > (float)0)
          {
            count++;
            cloud->push_back(point);
          }
        }
      }
    }

    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = "base_link";
    cloud->header = pcl_conversions::toPCL(cloud_msg.header);
  }

  void cloud_cb(const sensor_msgs::PointCloud2 &input)
  {
    unsigned char *indata = (unsigned char *)&input.data[0];
    int32_t num = 20185235;
    void *input_arr = malloc(num * sizeof(float));
    // char* fmt = sensor_msgs::PointCloud2::get_struct_fmt(input.is_bigendian, input.fields);
    // printf("format is: %d \n", (input.fields[3].offset));
    processBEV(input_arr, indata, input.height, input.width, input.point_step, input.row_step);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZI>);
    roi_bbox_gen(input_arr, cloud1);
    pt_cloud_pub(input_arr, cloud2);
    pub1.publish(cloud1);
    pub2.publish(cloud2);
    int roiNum = 0;
    // boost::random::mt19937 randgen;
    // boost::random::uniform_int_distribution<int> x_dist(1, 699);
    // boost::random::uniform_int_distribution<int> y_dist(1, 799);
    // boost::random::uniform_int_distribution<int> z_dist(1, 34);
    // bool *idx_done = (bool *)malloc(num * sizeof(bool));
    // //
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::PointXYZI point;
    // //
    // radial_lim = 50;
    // //
    // while (roiNum <= 20)
    // {
    //   //Generate a random initial point
    //   x = x_dist(randgen);
    //   y = y_dist(randgen);
    //   z = z_dist(randgen);
    //   //
    //   if (*(idx_done + at3(y, x, z)) != 1)
    //   {
    //     //If coordinates have not been read do:
    //     //
    //     //See if coordinates contain a valid lidar point
    //     is_point = *(((float *)input_arr) + at3(y, x, z));
    //     if (is_point == 1)
    //     {
    //       //Read all valid points in an ROI within radius of 5
    //       //
    //       pts_in_roi = 0;
    //       //
    //       for (r = 0; r <= radial_lim; r++)
    //       {
    //         for (th = 0; th <= 2 * M_PI; th += 0.1 * M_PI)
    //         {
    //           for (phi = 0; phi <= 2 * M_PI; phi += 0.1 * M_PI)
    //           {
    //             x_plus = r * sin(phi) * cos(th);
    //             y_plus = r * sin(phi) * sin(th);
    //             z_plus = r * cos(phi);
    //             if (x + x_plus <= 700 && y + y_plus <= 800 && z + z_plus <= 35)
    //             {
    //               is_point = *(((float *)input_arr) + at3(y + y_plus, x + x_plus, z + z_plus));
    //               if (is_point == 1)
    //               {
    //                 printf("x: %i , y: %i , z: %i \n", x_plus, y_plus, z_plus);
    //                 pts_in_roi++;
    //               }
    //             }
    //           }
    //         }
    //       }
    //       //
    //       if (pts_in_roi > 10)
    //       {
    //         //If roi dense enough, i.e. num of valid pts >= 20 do:
    //         printf("roi found, num of points: %i \n", pts_in_roi);
    //         point.x = radial_lim + x;
    //         point.y = radial_lim + y;
    //         point.z = x - radial_lim;
    //         point.intensity = y - radial_lim;
    //         cloud->push_back(point);
    //         roiNum++;
    //       }
    //       else
    //       {
    //         //Show that point is scanned but no valid ROI
    //         *(idx_done + at3(y, x, z)) = 2;
    //       }
    //     }
    //     //Show that point is scanned with a valid ROI
    //     *(idx_done + at3(y, x, z)) = 1;
    //   }
    // }
    // sensor_msgs::PointCloud2 cloud_msg;
    // cloud_msg.header.frame_id = "base_link";
    // cloud->header = pcl_conversions::toPCL(cloud_msg.header);
    // pub2.publish(cloud);
    // free(idx_done);
    free(input_arr);
  }

protected:
  ros::NodeHandle nh;
  ros::Publisher pub1, pub2;
  ros::Subscriber sub;
  int x, y, z, i, j, k;
  int count, pt_count;
  const int winsize = 50;
  float inten;
  int x_plus, y_plus, z_plus;
  float is_point;
  int r, th, phi;
  int pts_in_roi;
  int radial_lim;
};

int main(int argc, char **argv)
{
  // Initialize ROS
  ros::init(argc, argv, "PoitsCloudDeal");
  cloudHandler handler;
  ros::spin();
  return 0;
}
