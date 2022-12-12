// STL
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <netinet/in.h>
#include <sys/types.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
// SFML
// #include <SFML/Graphics.hpp>
// #include <SFML/OpenGL.hpp>
// My includes
#include "gmm_coverage/FortuneAlgorithm.h"
#include "gmm_coverage/Voronoi.h"
// #include "Graphics.h"
// ROS includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "sensor_msgs/msg/channel_float32.hpp"
#include "std_msgs/msg/int16.hpp"
#include "std_msgs/msg/bool.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "turtlebot3_msgs/msg/gaussian.hpp"
#include "turtlebot3_msgs/msg/gmm.hpp"

#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;




class Visualizer : public rclcpp::Node
{

public:
    Visualizer() : Node("gmm_visualizer")
    {
        SHAPE = visualization_msgs::msg::Marker::SPHERE;

        // --------------------------------------------------------- GMM ROS publishers and subscribers -------------------------------------------------------
        publisher = this->create_publisher<visualization_msgs::msg::MarkerArray>("/gmm_markers", 1);
        timer_ = this->create_wall_timer(1000ms, std::bind(&Visualizer::show, this));
        gmmSub_ = this->create_subscription<turtlebot3_msgs::msg::GMM>("/gaussian_mixture_model", 1, std::bind(&Visualizer::gmm_callback, this, _1));

    }
    ~Visualizer()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    void gmm_callback(const turtlebot3_msgs::msg::GMM::SharedPtr msg);
    void GroundColorMix(double* color, double x, double min, double max);
    void show();



private:

    uint32_t SHAPE;
    std::vector<std::vector<float>> MEANs;
    std::vector<std::vector<std::vector<float>>> VARs;
    std::vector<float> weights;

    //------------------------- Publishers and subscribers ------------------------------
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher;
    rclcpp::Subscription<turtlebot3_msgs::msg::GMM>::SharedPtr gmmSub_;
    turtlebot3_msgs::msg::GMM gmm_msg;
    visualization_msgs::msg::MarkerArray markers_msg;
    std::vector<visualization_msgs::msg::Marker> markers_array;
    //-----------------------------------------------------------------------------------

};

void Visualizer::gmm_callback(const turtlebot3_msgs::msg::GMM::SharedPtr msg)
{
    this->gmm_msg.gaussians = msg->gaussians;
    this->gmm_msg.weights = msg->weights;
}

// Returns RGB color in colormap from green to red for a given value x
void Visualizer::GroundColorMix(double* color, double x, double min = 0.0, double max = 1.0)
{
   /*  */
    double posSlope = (max-min)/60;
    double negSlope = (min-max)/60;

    if( x < 60 )
    {
        color[0] = max;
        color[1] = posSlope*x+min;
        color[2] = min;
        return;
    }
    else if ( x < 120 )
    {
        color[0] = negSlope*x+2*max+min;
        color[1] = max;
        color[2] = min;
        return;
    }
    else if ( x < 180  )
    {
        color[0] = min;
        color[1] = max;
        color[2] = posSlope*x-2*max+min;
        return;
    }
    else if ( x < 240  )
    {
        color[0] = min;
        color[1] = negSlope*x+4*max+min;
        color[2] = max;
        return;
    }
    else if ( x < 300  )
    {
        color[0] = posSlope*x-4*max+min;
        color[1] = min;
        color[2] = max;
        return;
    }
    else
    {
        color[0] = max;
        color[1] = min;
        color[2] = negSlope*x+6*max;
        return;
    }
}

void Visualizer::show()
{

    for (int i = 0; i < this->gmm_msg.gaussians.size(); i++)
    {
        // -------------- Draw ellipse from covariance matrix ----------------
        // Elements of the covariance matrix
        double a = this->gmm_msg.gaussians[i].covariance[0];
        double b = this->gmm_msg.gaussians[i].covariance[1];
        double c = this->gmm_msg.gaussians[i].covariance[3];

        // Ellipse's parameters
        double lambda1 = (a+c)/2 + sqrt( pow((a-c)/2,2) + pow(b,2) );
        double lambda2 = (a+c)/2 - sqrt( pow((a-c)/2,2) + pow(b,2) );
        double r1 = sqrt(lambda1);
        double r2 = sqrt(lambda2);
        double theta = 0.0;
        if (b == 0.0 && a >= c)
            theta = 0.0;
        else if (b == 0.0 && a < c)
            theta = M_PI/2;
        else
            theta = atan2((lambda1-a), b);
        visualization_msgs::msg::Marker m;
        m.id = i;
        m.header.frame_id = "odom";
        m.type = SHAPE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = this->gmm_msg.gaussians[i].mean_point.x;
        m.pose.position.y = this->gmm_msg.gaussians[i].mean_point.y;
        m.pose.position.z = this->gmm_msg.gaussians[i].mean_point.z;
        if (this->gmm_msg.gaussians[i].covariance.size() == 9)
        {
            // 3D case
            m.scale.x = this->gmm_msg.gaussians[i].covariance[0];
            m.scale.y = this->gmm_msg.gaussians[i].covariance[4];
            m.scale.z = this->gmm_msg.gaussians[i].covariance[8];
        } else
        {   
            // 2D case
            // m.scale.x = this->gmm_msg.gaussians[i].covariance[0];
            // m.scale.y = this->gmm_msg.gaussians[i].covariance[3];
            m.scale.x = r1;
            m.scale.y = r2;
            m.scale.z = 0.01;
            tf2::Quaternion q;
            q.setRPY(0, 0, theta);
            m.pose.orientation.x = q.x();
            m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z();
            m.pose.orientation.w = q.w();
        }
        // m.color.r = 0.0f;
        // m.color.g = 1.0f;
        // m.color.b = 0.0f;
        // m.color.a = this->gmm_msg.weights[i];           // alpha is given by its weight
        double color[3];
        GroundColorMix(color, 360.0 * this->gmm_msg.weights[i]);
        m.color.r = color[0];
        m.color.g = color[1];
        m.color.b = color[2];
        m.color.a = 1.0;
        this->markers_msg.markers.push_back(m);
    }
    this->publisher->publish(this->markers_msg);
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Visualizer>());
    rclcpp::shutdown();
    return 0;
}
