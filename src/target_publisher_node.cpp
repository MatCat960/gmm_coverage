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

// ROS includes
#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/pose.hpp>


using namespace std::chrono_literals;
using std::placeholders::_1;



class Target : public rclcpp::Node
{

public:
    Target() : Node("target_publisher_node")
    {
        //------------------------------------------------- ROS parameters ---------------------------------------------------------
        this->declare_parameter<float>("XT", 0.0);
        this->get_parameter("XT", XT);
        this->declare_parameter<float>("YT", 0.0);
        this->get_parameter("YT", YT);
        this->declare_parameter<int>("ID", 0);
        this->get_parameter("ID", ID);
        
        //-----------------------------------------------------------------------------------------------------------------------------------

        //--------------------------------------------------- Subscribers and Publishers ----------------------------------------------------
        pub_ = this->create_publisher<geometry_msgs::msg::Pose>("/target" + std::to_string(ID) + "/pose", 1);
        timer_ = this->create_wall_timer(100ms, std::bind(&Target::loop, this));

        msg.position.x = XT;
        msg.position.y = YT;
        msg.position.z = 0.0;

        std::cout << "Target " << ID << " created in position " << XT << " " << YT << std::endl;

        
    }
    ~Target()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    //void stop(int signum);
    void loop();

private:
    double XT;
    double YT;
    int ID;

    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    geometry_msgs::msg::Pose msg;
};

void Target::loop()
{
    pub_->publish(msg);
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Target>());
    rclcpp::shutdown();
    return 0;
}