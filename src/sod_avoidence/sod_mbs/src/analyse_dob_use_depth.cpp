#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "MBS.hpp" 
#include <chrono>
#include <limits>
#include <fstream>
#include <json/json.h>
#include <signal.h>

struct Pixel {
    int x, y;
};

struct Obstacle {
    std::vector<Pixel> pixels;
    std::vector<pcl::PointXYZ> points3D;
    double risk;
};

// 相机参数
const double camera_factor = 1000;
const double camera_fx = 462.138;
const double camera_fy = 462.138;
const double camera_cx = 320.0;
const double camera_cy = 240.0;

std::vector<Obstacle> detectSalientObstacles(const cv::Mat& img, double& detection_time) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat saliencyMap = doWork(img, true, true, false); 
    
    auto end = std::chrono::high_resolution_clock::now();
    detection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::vector<Obstacle> obstacles;

    if (!saliencyMap.empty()) {
        if (saliencyMap.depth() != CV_8U) {
            saliencyMap.convertTo(saliencyMap, CV_8UC1, 255.0);
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(saliencyMap, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        obstacles.reserve(contours.size());
        cv::Mat mask = cv::Mat::zeros(saliencyMap.size(), CV_8UC1);
        
        for (const auto& contour : contours) {
            mask.setTo(cv::Scalar(0));
            cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{contour}, cv::Scalar(255));

            Obstacle obstacle;
            cv::Mat nonZeroCoordinates;
            cv::findNonZero(mask, nonZeroCoordinates);

            obstacle.pixels.reserve(nonZeroCoordinates.total());
            for (int i = 0; i < nonZeroCoordinates.total(); ++i) {
                const auto& point = nonZeroCoordinates.at<cv::Point>(i);
                obstacle.pixels.emplace_back(Pixel{point.x, point.y});
            }
            obstacles.emplace_back(std::move(obstacle));
        }
    }
    return obstacles;
}

bool isDepthConsistent(const cv::Mat& depthImage, int x, int y, float depth, float threshold) {
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float neighborDepth = depthImage.at<float>(y + dy, x + dx);
            if (std::abs(depth - neighborDepth) > threshold) {
                return false;
            }
        }
    }
    return true;
}

pcl::PointXYZ convertPixelToPoint(int x, int y, double Z) {
    Z /= camera_factor; // mm转换为m
    return {
        static_cast<float>((x - camera_cx) * Z / camera_fx),
        static_cast<float>((y - camera_cy) * Z / camera_fy),
        static_cast<float>(Z)
    };
}

void extractObstaclePoints(std::vector<Obstacle>& obstacles, const cv::Mat& depthImage) {
    for (auto& obstacle : obstacles) {
        obstacle.points3D.reserve(obstacle.pixels.size());
        for (const auto& pixel : obstacle.pixels) {
            float depth = depthImage.at<float>(pixel.y, pixel.x);

            if (pixel.x > 0 && pixel.x < depthImage.cols - 1 && 
                pixel.y > 0 && pixel.y < depthImage.rows - 1 &&
                isDepthConsistent(depthImage, pixel.x, pixel.y, depth, 20.f)) {
                obstacle.points3D.emplace_back(convertPixelToPoint(pixel.x, pixel.y, depth));
            }
        }
    }
}

void calculateObstacleRisks(std::vector<Obstacle>& obstacles) {
    const double characteristic_distance = 2.5;
    const double char_dist_squared = characteristic_distance * characteristic_distance;
    const double max_expected_size = 4000.0;

    for (auto& obstacle : obstacles) {
        if (obstacle.points3D.empty()) continue;

        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        for (const auto& point : obstacle.points3D) {
            center += point.getVector3fMap();
        }

        float size = obstacle.points3D.size();
        center /= size;
        double centerDist = center.norm();

        double sizerisk = std::min(1.0, size / max_expected_size);

        Eigen::Vector3f robotDirection(0, 0, 1);
        double dotProduct = robotDirection.dot(center.normalized());
        double angle = std::acos(std::abs(dotProduct))/M_PI;
        double directionRisk = std::exp(-angle * angle / (2 * 0.5 * 0.5));

        double distancerisk = 1.0 / (1.0 + (centerDist * centerDist) / char_dist_squared);

        obstacle.risk = sizerisk * directionRisk * distancerisk;
    }
}

void displayDangerousObstacles(const cv::Mat& img, const std::vector<Obstacle>& obstacles) {
    cv::Mat displayImage = cv::Mat::zeros(img.size(), img.type());
    for (const auto& obstacle : obstacles) {
        int brightness = static_cast<int>(obstacle.risk * 255.0);
        brightness = std::min(255, std::max(0, brightness));
        
        for (const auto& pixel : obstacle.pixels) {
            displayImage.at<cv::Vec3b>(pixel.y, pixel.x) = cv::Vec3b(brightness, brightness, brightness);
        }
    }
    cv::namedWindow("Dangerous Obstacles Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("Dangerous Obstacles Map", 480, 320);
    cv::imshow("Dangerous Obstacles Map", displayImage);
    cv::waitKey(20);
}

void PublishDangerousObstacleCloud(
    const std::vector<Obstacle>& obstacles,
    ros::Publisher& publisher,
    const std::string& frame_id,
    const ros::Time& timestamp,
    double riskThreshold = 0.1)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr dangerousCloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto& obstacle : obstacles) {
        if (obstacle.risk > riskThreshold) {
            dangerousCloud->insert(dangerousCloud->end(), obstacle.points3D.begin(), obstacle.points3D.end());
        }
    }

    dangerousCloud->width = dangerousCloud->points.size();
    dangerousCloud->height = 1;
    dangerousCloud->is_dense = true;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*dangerousCloud, cloud_msg);
    
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = timestamp;
    publisher.publish(cloud_msg);
}

class ObstacleDetectionNode
{
public:
    ObstacleDetectionNode(ros::NodeHandle& nh)
        : point_cloud_publisher_(nh.advertise<sensor_msgs::PointCloud2>("saliency_point_cloud", 5)),
          detection_time_publisher_(nh.advertise<std_msgs::Float64>("saliency_detection_time", 5)),
          total_detection_time_(0),
          detection_count_(0),
          max_detection_time_(std::numeric_limits<double>::min()),
          min_detection_time_(std::numeric_limits<double>::max())
    {
        rgb_sub_.subscribe(nh, "/D435i_camera/color/image_raw", 5);
        depth_sub_.subscribe(nh, "/D435i_camera/aligned_depth_to_color/image_raw", 5);

        sync_.reset(new Synchronizer(SyncPolicy(10), rgb_sub_, depth_sub_));
        sync_->registerCallback(boost::bind(&ObstacleDetectionNode::imageCallback, this, _1, _2));
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg)
    {
        cv_bridge::CvImagePtr cv_ptr_rgb, cv_ptr_depth;
        try
        {
            cv_ptr_rgb = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
            cv_ptr_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        double detection_time;
        auto obstacles = detectSalientObstacles(cv_ptr_rgb->image, detection_time);
        
        // Update statistics
        total_detection_time_ += detection_time;
        detection_count_++;
        max_detection_time_ = std::max(max_detection_time_, detection_time);
        min_detection_time_ = std::min(min_detection_time_, detection_time);

        // Calculate and publish average detection time
        double avg_detection_time = total_detection_time_ / detection_count_;
        std_msgs::Float64 time_msg;
        time_msg.data = avg_detection_time;
        detection_time_publisher_.publish(time_msg);

        extractObstaclePoints(obstacles, cv_ptr_depth->image);
        calculateObstacleRisks(obstacles);
        displayDangerousObstacles(cv_ptr_rgb->image, obstacles);
        PublishDangerousObstacleCloud(obstacles, point_cloud_publisher_, depth_msg->header.frame_id, depth_msg->header.stamp);
    }

    void saveStatisticsToJson()
    {
        Json::Value root;
        root["average_detection_time_ms"] = total_detection_time_ / detection_count_;
        root["max_detection_time_ms"] = max_detection_time_;
        root["min_detection_time_ms"] = min_detection_time_;
        root["total_detections"] = detection_count_;

        Json::StreamWriterBuilder writer;
        
        // Specify the full path for the JSON file
        std::string json_file = "/home/qun/turtlebot3_realsensed435i/saliency_detection_statistics.json";
        std::ofstream file_id(json_file);
        
        if (file_id.is_open())
        {
            std::unique_ptr<Json::StreamWriter> jsonWriter(writer.newStreamWriter());
            jsonWriter->write(root, &file_id);
            file_id.close();
            ROS_INFO("Statistics saved to %s", json_file.c_str());
        }
        else
        {
            ROS_ERROR("Unable to open file for writing statistics: %s", json_file.c_str());
        }
    }

private:
    ros::Publisher point_cloud_publisher_;
    ros::Publisher detection_time_publisher_;
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
    boost::shared_ptr<Synchronizer> sync_;

    double total_detection_time_;
    int detection_count_;
    double max_detection_time_;
    double min_detection_time_;
};

ObstacleDetectionNode* node_ptr = nullptr;

void signalHandler(int signum)
{
    ROS_INFO("Interrupt signal (%d) received.", signum);
    if (node_ptr)
    {
        node_ptr->saveStatisticsToJson();
    }
    ros::shutdown();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "Dangerous_Obstacle_to_Point_Cloud", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh;

    ObstacleDetectionNode obstacleDetectionNode(nh);
    node_ptr = &obstacleDetectionNode;

    // Register custom signal handler
    signal(SIGINT, signalHandler);

    ros::spin();
    return 0;
}