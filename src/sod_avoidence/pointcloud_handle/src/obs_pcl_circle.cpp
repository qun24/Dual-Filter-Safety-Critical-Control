#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense>
#include <vector>
#include <sstream>
#include <cmath>
#include <visualization_msgs/Marker.h>
#include <omp.h>

ros::Publisher obstacle_info_pub;
ros::Publisher marker_pub;

double x_real = 0;
double y_real = 0;
double theta_real = 0;

// Define camera offset relative to the vehicle
const double camera_offset_x = 0.0;  
const double camera_offset_y = 0.0; 

void newOdomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    x_real = msg->pose.pose.position.x;
    y_real = msg->pose.pose.position.y;

    tf::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch;
    m.getRPY(roll, pitch, theta_real);
}

void publishMarker(ros::Publisher& marker_pub, const Eigen::Vector2f& position, int id, double r, double g, double b) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "odom";
    marker.header.stamp = ros::Time::now();
    marker.ns = "obstacle";
    marker.id = id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = position.x();
    marker.pose.position.y = position.y();
    marker.pose.position.z = 0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.a = 1.0;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker_pub.publish(marker);
}

// Calculate centroid
Eigen::Vector2f calculateCentroid(const std::vector<Eigen::Vector2f>& points) {
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (const auto& point : points) {
        centroid += point;
    }
    return centroid / static_cast<double>(points.size());
}

// Find the point farthest from the centroid
Eigen::Vector2f findFarthestPoint(const std::vector<Eigen::Vector2f>& points, const Eigen::Vector2f& centroid) {
    Eigen::Vector2f farthest_point;
    double max_distance = 0;
    for (const auto& point : points) {
        double distance = (point - centroid).norm();
        if (distance > max_distance) {
            max_distance = distance;
            farthest_point = point;
        }
    }
    return farthest_point;
}

std::pair<Eigen::Vector2f, double> calculateMinEnclosingCircle(const std::vector<pcl::PointXYZ>& points, const Eigen::Vector2f& pn) {
    // Project point cloud information onto the xy plane
    std::vector<Eigen::Vector2f> xy_projected_points;
    xy_projected_points.reserve(points.size());
    for (const auto& point : points) {
        xy_projected_points.emplace_back(point.x, point.y);
    }
    // centroid as pg, Farthest_point as pm, and sensor_origin as pn
    Eigen::Vector2f pg = calculateCentroid(xy_projected_points);
    Eigen::Vector2f pm = findFarthestPoint(xy_projected_points, pg); // Find the projected point farthest from the centroid
    
    // Calculate distances
    double pm_pn = (pm - pn).norm();
    double pg_pn = (pg - pn).norm();
    double pm_pg = (pm - pg).norm();

    double alpha = std::acos((std::pow(pm_pn, 2) + std::pow(pg_pn, 2) - std::pow(pm_pg, 2)) / (2 * pm_pn * pg_pn));
    double beta = M_PI / 2 - alpha;
    double pc_pm = pm_pn * std::tan(alpha);
    double pc_pg = pc_pm * std::sin(beta) / beta;
    double pc_pn = pc_pg + pg_pn;
    // Calculate the vector from pn to pg (pg_pn)
    Eigen::Vector2f pg_pn_vector = pg - pn;
    // Calculate the position of pc
    Eigen::Vector2f pc_vector = pn + pg_pn_vector.normalized() * pc_pn;
    double radius = (pm - pc_vector).norm(); // Calculate radius

    // publishMarker(marker_pub, pg, 0, 0.0, 1.0, 0.0); // Green // Publish centroid
    // publishMarker(marker_pub, pm, 1, 1.0, 0.0, 0.0); // Red // Publish farthest point
    // publishMarker(marker_pub, pc_vector, 2, 0.0, 0.0, 1.0); // Blue // Publish circle center
    
    // Return the center and radius of the minimum enclosing circle
    return std::make_pair(pc_vector, radius);
}

void publishObstacleInfo(const std::string& info) {
    std_msgs::String obstacle_info_msg;
    obstacle_info_msg.data = info;
    obstacle_info_pub.publish(obstacle_info_msg);
}

std::string joinStrings(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) return "";
    std::ostringstream oss;
    std::copy(strings.begin(), strings.end() - 1, std::ostream_iterator<std::string>(oss, delimiter.c_str()));
    oss << strings.back();
    return oss.str();
}

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->empty()) {
        ROS_WARN("Received an empty point cloud. Skipping processing.");
        publishObstacleInfo("");
        return;
    }
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.25); // Points with distance <= 0.25m will be considered in the same cluster
    ec.setMinClusterSize(5);
    ec.setMaxClusterSize(20000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    
    std::vector<std::string> obstacle_info;

    // Calculate camera position pn
    Eigen::Vector2f pn = Eigen::Vector2f(x_real + camera_offset_x, y_real + camera_offset_y); 

    #pragma omp parallel for
    for (const auto& indices : cluster_indices) {
        if (indices.indices.size() < 3) {
            continue; // Skip noise clusters
        }

        std::vector<pcl::PointXYZ> cluster;
        for (const auto& index : indices.indices) {
            cluster.push_back((*cloud)[index]);
        }
        
        auto [center, radius] = calculateMinEnclosingCircle(cluster, pn);

        obstacle_info.emplace_back("position: (" + std::to_string(center.x()) + ", " + 
                                   std::to_string(center.y()) + "), radius: " + 
                                   std::to_string(radius));
        #pragma omp critical
    }
    std::string obstacle_info_msg_data = joinStrings(obstacle_info, "\n");
    publishObstacleInfo(obstacle_info_msg_data);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_processor");
    ros::NodeHandle nh;

    obstacle_info_pub = nh.advertise<std_msgs::String>("/obstacle_info", 5);
    marker_pub = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 10);
    ros::Subscriber odom_sub = nh.subscribe("/odom", 10, newOdomCallback);
    ros::Subscriber pointcloud_sub = nh.subscribe("Filtered_Dangerous_Obstacle_cloud", 5, pointCloudCallback);

    ros::spin();
    return 0;
}