#include <ros/ros.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <regex>
#include <set>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>

ros::Publisher marker_pub;
std::set<int> last_marker_ids;

void deleteMarker(int marker_id) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "odom";
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.id = marker_id;
    marker_pub.publish(marker);
}

std::vector<std::vector<float>> extractAndFormatData(const std::string& text) {
    std::regex re("[-+]?\\d*\\.\\d+|\\d+");
    std::sregex_iterator next(text.begin(), text.end(), re);
    std::sregex_iterator end;
    std::vector<float> numbers;

    while (next != end) {
        std::smatch match = *next;
        numbers.push_back(std::stof(match.str()));
        next++;
    }

    std::vector<std::vector<float>> formatted_data;
    for (size_t i = 0; i < numbers.size(); i += 5) {
        std::vector<float> obstacle_data = { 
            round(numbers[i] * 100.0f) / 100.0f, 
            round(numbers[i + 1] * 100.0f) / 100.0f, 
            round(numbers[i + 2] * 100.0f) / 100.0f,
            round(numbers[i + 3] * 100.0f) / 100.0f,
            round(numbers[i + 4] * 100.0f) / 100.0f
        };
        formatted_data.push_back(obstacle_data);
    }

    return formatted_data;
}

void drawSpheres(const std::vector<std::vector<float>>& formatted_data, std::set<int>& new_marker_ids) {
    for (size_t i = 0; i < formatted_data.size(); ++i) {
        const auto& obstacle = formatted_data[i];
        visualization_msgs::Marker marker;
        marker.header.frame_id = "odom";
        marker.type = visualization_msgs::Marker::SPHERE; 
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = marker.scale.y = marker.scale.z = obstacle[2] * 2; 
        marker.color.a = 1.0f;
        marker.pose.orientation.w = 1.0f;
        marker.pose.position.x = obstacle[0]; 
        marker.pose.position.y = obstacle[1]; 
        marker.pose.position.z = obstacle[2]; 
        marker.id = static_cast<int>(i);


        if (obstacle[3] == 0 && obstacle[4] == 0) {
            marker.color.g = 1.0f;
        } else {
            marker.color.r = 1.0f; 
            marker_pub.publish(marker);
            new_marker_ids.insert(static_cast<int>(i));

            for (int j = 1; j <= 3; ++j) {
                marker.pose.position.x += obstacle[3] * 0.3f;
                marker.pose.position.y += obstacle[4] * 0.3f;
                marker.color.r = 1.0f;
                marker.color.g = 1.0f;
                marker.color.b = 0.0f;
                marker.color.a = 0.5f;
                marker.id = static_cast<int>(i) + j * 10;
                marker.type = visualization_msgs::Marker::SPHERE;
                marker.action = visualization_msgs::Marker::ADD;
                new_marker_ids.insert(marker.id);
                marker_pub.publish(marker);
            }
            continue;
        }

        marker_pub.publish(marker);
        new_marker_ids.insert(static_cast<int>(i));
    }
}

void obstacleInfoCallback(const std_msgs::String::ConstPtr& msg) {
    std::string obstacle_info = msg->data;
    std::set<int> new_marker_ids;

    std::vector<std::vector<float>> formatted_list = extractAndFormatData(obstacle_info);
    drawSpheres(formatted_list, new_marker_ids); 

    for (const int marker_id : last_marker_ids) {
        if (new_marker_ids.find(marker_id) == new_marker_ids.end()) {
            deleteMarker(marker_id);
        }
    }

    last_marker_ids = new_marker_ids;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "obstacle_visualizer");
    ros::NodeHandle nh;

    marker_pub = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 10);
    ros::Subscriber sub = nh.subscribe("/processed_obstacles", 10, obstacleInfoCallback);

    ros::spin();
    return 0;
}