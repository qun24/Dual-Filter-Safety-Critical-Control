#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h> 
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

ros::Time last_received_time;
ros::Time empty_cloud_start_time;
bool sending_empty_clouds = false;
ros::Publisher pub;

void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg, const ros::Publisher& pub, tf2_ros::Buffer& tfBuffer) {
    if (cloud_msg->data.empty() || cloud_msg->width * cloud_msg->height == 0) {
    ROS_WARN("Received an empty point cloud. Skipping processing.");
    return;}

    last_received_time = ros::Time::now();  // Update the time of last received point cloud
    sending_empty_clouds = false;

    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    // ROS_INFO("Original cloud size: %lu points", cloud->size()); // Output the number of points in the original cloud
    
    // Apply Distance filter (removes points too far from the camera)
    pcl::PassThrough<pcl::PointXYZ> pass_distance;
    pass_distance.setInputCloud(cloud);
    pass_distance.setFilterFieldName("z");
    pass_distance.setFilterLimits(0, 3.5); // Keep points within 3.5 meters of the camera
    PointCloud::Ptr cloud_filtered_distance(new PointCloud);
    pass_distance.filter(*cloud_filtered_distance);
    // ROS_INFO("After distance filter (z ≤ 4.0 m): %lu points", cloud_filtered_distance->size()); // Output the number of points after distance filtering

    // Apply VoxelGrid filter
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud_filtered_distance);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);  // Adjust leaf size as needed
    PointCloud::Ptr cloud_voxel_filtered(new PointCloud);
    vg.filter(*cloud_voxel_filtered);
    // ROS_INFO("After VoxelGrid filter: %lu points", cloud_voxel_filtered->size()); // Output the number of points after voxel grid filtering

    // Apply Uniform Sampling filter
    // pcl::UniformSampling<pcl::PointXYZ> us;
    // us.setInputCloud(cloud_filtered_distance);
    // us.setRadiusSearch(0.05f);  // Set search radius
    // PointCloud::Ptr cloud_uni_filtered(new PointCloud);
    // us.filter(*cloud_uni_filtered);

    // Initialize RadiusOutlierRemoval filter
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(cloud_voxel_filtered);
    // ror.setInputCloud(cloud_uni_filtered);
    ror.setRadiusSearch(0.3);  // Set search radius to 0.3 meters
    ror.setMinNeighborsInRadius(10);  // Set minimum number of neighbors to 10
    PointCloud::Ptr cloud_outlier_filtered(new PointCloud);
    ror.filter(*cloud_outlier_filtered);

    // Get the transform
    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tfBuffer.lookupTransform("odom", cloud_msg->header.frame_id, cloud_msg->header.stamp, ros::Duration(0.1));
    } catch (tf2::TransformException &ex) {
        ROS_WARN("%s", ex.what());
        return;
    }

    // Apply the transform directly to the filtered point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f transform;
    pcl_ros::transformAsMatrix(transformStamped.transform, transform);
    pcl::transformPointCloud(*cloud_outlier_filtered, *cloud_transformed, transform);

    // Apply PassThrough filter based on odom z-axis
    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(cloud_transformed);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(0.02, 4.0); // Set z-axis filter range to remove ground points, this step must not be modified
    PointCloud::Ptr cloud_global_filtered(new PointCloud);
    pass_z.filter(*cloud_global_filtered); // Store the filtered point cloud in temp_cloud
    // ROS_INFO("After z-axis PassThrough filter: %lu points", cloud_z_filtered->size()); // Output the number of points after z-axis filtering

    // Publish the filtered point cloud
    sensor_msgs::PointCloud2 cloud_output;
    pcl::toROSMsg(*cloud_global_filtered, cloud_output);
    cloud_output.header.frame_id = "odom"; // Transformed coordinate frame
    cloud_output.header.stamp = cloud_msg->header.stamp; // Use the original timestamp
    pub.publish(cloud_output);
}

void publish_empty_cloud() {
    ros::Time now = ros::Time::now();
    sensor_msgs::PointCloud2 empty_cloud;
    empty_cloud.header.frame_id = "odom";
    empty_cloud.header.stamp = now;
    pub.publish(empty_cloud);
    // ROS_INFO("Published empty cloud");
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_filter");
    ros::NodeHandle nh;

    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    pub = nh.advertise<sensor_msgs::PointCloud2>("Filtered_Dangerous_Obstacle_cloud", 10);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("saliency_point_cloud", 10, boost::bind(&cloud_callback, _1, pub, boost::ref(tfBuffer)));
    ros::Rate rate(25); // 25 Hz
    while (ros::ok()) {
        ros::spinOnce();
        ros::Time now = ros::Time::now();
        ros::Duration time_since_last_received = now - last_received_time;

        if (time_since_last_received.toSec() > 0.5 && !sending_empty_clouds) {
            // If no message has been received, start sending empty point clouds immediately
            empty_cloud_start_time = now;
            sending_empty_clouds = true;
        }

        if (sending_empty_clouds) {
            ros::Duration sending_duration = ros::Time::now()  - empty_cloud_start_time;
            if (sending_duration.toSec() < 5.0) {
                publish_empty_cloud();
            } else {
                sending_empty_clouds = false;
            }
        }

        rate.sleep();
    }
    // ros::spin();
}