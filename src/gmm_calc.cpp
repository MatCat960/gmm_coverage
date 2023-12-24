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

// My includes
#include "gmm_coverage/Graphics.h"

// ROS includes
#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "turtlebot3_msgs/msg/gaussian.hpp"
#include "turtlebot3_msgs/msg/gmm.hpp"

#include "gaussian_mixture_model/gaussian_mixture_model.h"



using namespace std::chrono_literals;
using std::placeholders::_1;



class Supervisor : public rclcpp::Node
{

public:
    Supervisor() : Node("gmm_calc"), gmm(2)
    {
        //------------------------------------------------- ROS parameters ---------------------------------------------------------
        this->declare_parameter<int>("ID", 0);
        this->get_parameter("ID", ID);
        this->declare_parameter<int>("ROBOTS_NUM", 8);
        this->get_parameter("ROBOTS_NUM", ROBOTS_NUM);
        this->declare_parameter<int>("NUM_SAMPLES", 500);
        this->get_parameter("NUM_SAMPLES", NUM_SAMPLES);
        this->declare_parameter<int>("TARGETS_NUM", 2);
        this->get_parameter("TARGETS_NUM", TARGETS_NUM);
        this->declare_parameter<double>("SENS_RANGE", 5.0);
        this->get_parameter("SENS_RANGE", SENS_RANGE);
        this->declare_parameter<double>("COMM_RANGE", 5.0);
        this->get_parameter("COMM_RANGE", COMM_RANGE);
        this->declare_parameter<double>("ENV_SIZE", 20.0);
        this->get_parameter("ENV_SIZE", ENV_SIZE);
        this->declare_parameter<bool>("GRAPHICS_ON", true);
        this->get_parameter("GRAPHICS_ON", GRAPHICS_ON);
        //-----------------------------------------------------------------------------------------------------------------------------------

        //------------------------------------------------- ROS publishers and subscribers -------------------------------------------------
        gmm_pub_ = this->create_publisher<turtlebot3_msgs::msg::GMM>("/gaussian_mixture_model_"+std::to_string(ID), 10);
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("turtlebot" + std::to_string(ID) + "/odom", 1, std::bind(&Supervisor::odomCallback, this, _1));
        for (int i = 0; i < TARGETS_NUM; i++)
        {
            target_subs_.push_back(this->create_subscription<geometry_msgs::msg::Pose>("target" + std::to_string(i) + "/pose", 1, [this, i](geometry_msgs::msg::Pose::SharedPtr msg) {this->targetCallback(msg,i);}));
        }
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            if (i != ID)
            {
                neighbor_subs_.push_back(this->create_subscription<nav_msgs::msg::Odometry>("turtlebot" + std::to_string(i) + "/odom", 1, [this, i](nav_msgs::msg::Odometry::SharedPtr msg) {this->neighCallback(msg,i);}));
            }
        }
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            if (i != ID)
            {
                communication_subs_.push_back(this->create_subscription<geometry_msgs::msg::Point>("robot" + std::to_string(i) + "/communication_topic", 1, [this, i](geometry_msgs::msg::Point::SharedPtr msg) {this->communicationCallback(msg,i);}));
            }
        }
        communication_pub_ = this->create_publisher<geometry_msgs::msg::Point>("robot" + std::to_string(ID) + "/communication_topic", 1);
        timer_ = this->create_wall_timer(2000ms, std::bind(&Supervisor::loop, this));

        // ------------------------------------------------- Initialize GMM and samples -----------------------------------------------------
        // Set up random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-0.5*ENV_SIZE, 0.5*ENV_SIZE);
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            Eigen::VectorXd point(2);
            double x = dis(gen);
            double y = dis(gen);
            point << x, y;
            samples.push_back(point);
        }

        // Initialize real targets position
        for (int i = 0; i < TARGETS_NUM; i++)
        {
            Eigen::VectorXd point(2);
            point << 0.0, 0.0;
            targets_real.push_back(point);
        }

        // ---------------------------------------------- Initialize robot's pose ----------------------------------------------------------
        pose.resize(2);
        pose << 0.0, 0.0;

        // Neighbors poses
        p_j.resize(2, ROBOTS_NUM);
        p_j.setZero();

        //----------------------------------------------- Graphics window -----------------------------------------------
        if (GRAPHICS_ON)
        {
            app_gui.reset(new Graphics{ENV_SIZE, ENV_SIZE, -0.5*ENV_SIZE, -0.5*ENV_SIZE, 2.0});
        }

        

        
    }
    ~Supervisor()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
        if ((GRAPHICS_ON) && (this->app_gui->isOpen())){
            this->app_gui->close();
        }
    }

    //void stop(int signum);
    void loop();
    void targetCallback(const geometry_msgs::msg::Pose::SharedPtr msg, int i);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void neighCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int i);
    void communicationCallback(const geometry_msgs::msg::Point::SharedPtr msg, int i);
    Eigen::VectorXd addNoise(Eigen::VectorXd point);

private:
    int ID;
    int NUM_SAMPLES;
    int ROBOTS_NUM;
    int TARGETS_NUM;
    double COMM_RANGE;
    double SENS_RANGE;
    double ENV_SIZE;
    bool GRAPHICS_ON;
    GaussianMixtureModel gmm;
    std::vector<Eigen::VectorXd> samples;
    Eigen::VectorXd pose;
    Eigen::MatrixXd p_j;
    std::vector<Eigen::VectorXd> targets_real;

    rclcpp::Publisher<turtlebot3_msgs::msg::GMM>::SharedPtr gmm_pub_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr> target_subs_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> neighbor_subs_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr communication_pub_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr> communication_subs_;
    rclcpp::TimerBase::SharedPtr timer_;

    //Rendering with SFML
    //------------------------------ graphics window -------------------------------------
    std::unique_ptr<Graphics> app_gui;
    //------------------------------------------------------------------------------------
};

void Supervisor::targetCallback(const geometry_msgs::msg::Pose::SharedPtr msg, int i)
{
    Eigen::VectorXd point(2);
    point << msg->position.x, msg->position.y;

    // Update real target position
    targets_real[i] = point;

    // Add to the list only if inside sensing range
    double d = (point - pose).norm();
    if (d < SENS_RANGE)
    {
        // Add noise to the target position
        Eigen::VectorXd noisy_point(2);
        noisy_point = addNoise(point);

        // Remove oldest sample and add the new one
        samples.erase(samples.begin());
        samples.push_back(noisy_point);

        // Communicate detected point to neighbors
        geometry_msgs::msg::Point comm_msg;
        comm_msg.x = noisy_point(0);
        comm_msg.y = noisy_point(1);
        this->communication_pub_->publish(comm_msg);
    }
}

void Supervisor::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    pose << msg->pose.pose.position.x, msg->pose.pose.position.y;
}

void Supervisor::neighCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int i)
{
    p_j(0, i) = msg->pose.pose.position.x;
    p_j(1, i) = msg->pose.pose.position.y;
}

void Supervisor::communicationCallback(const geometry_msgs::msg::Point::SharedPtr msg, int i)
{
    Eigen::VectorXd point(2);
    point << msg->x, msg->y;

    // Add to the list only if inside communication range
    double d = (p_j.col(i) - pose).norm();
    if (d < COMM_RANGE)
    {
        // Remove oldest sample and add the new one
        samples.erase(samples.begin());
        samples.push_back(point);
    }
}

Eigen::VectorXd Supervisor::addNoise(Eigen::VectorXd point)
{
    // Generate random inputs
    std::random_device rd;
    std::mt19937 gen(rd());
    double dev = 1.0;
    std::normal_distribution<double> dist_x(point(0), dev);
    std::normal_distribution<double> dist_y(point(1), dev);

    Eigen::VectorXd noisy_point(2);
    noisy_point << dist_x(gen), dist_y(gen);

    
    return noisy_point;
}

void Supervisor::loop()
{
    auto timerstart = std::chrono::high_resolution_clock::now();
    gmm.fitgmm(samples, 2, 1000, 1e-3, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Computation time for EM: -------------: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - timerstart).count()<<" ms :-------------\n";

    std::vector<Eigen::VectorXd> mean_points = gmm.getMeans();
    std::vector<Eigen::MatrixXd> covariances = gmm.getCovariances();
    std::vector<double> weights = gmm.getWeights();

    // std::cout << "Mean Points: " << std::endl;
    // for(int i = 0; i < mean_points.size(); i++)
    // {
    //     std::cout << mean_points[i] << std::endl;
    // }

    // std::cout << "Covariances: " << std::endl;
    // for(int i = 0; i < covariances.size(); i++)
    // {
    //     std::cout << covariances[i] << std::endl;
    // }

    // std::cout << "Weights: " << std::endl;
    // for(int i = 0; i < weights.size(); i++)
    // {
    //     std::cout << weights[i] << std::endl;
    // }

    // Create ROS GMM msg
    turtlebot3_msgs::msg::GMM gmm_msg;
    gmm_msg.weights = weights;

    for(int i = 0; i < mean_points.size(); i++)
    {
        turtlebot3_msgs::msg::Gaussian gaussian;
        gaussian.mean_point.x = mean_points[i](0);
        gaussian.mean_point.y = mean_points[i](1);
        gaussian.covariance.push_back(covariances[i](0,0));
        gaussian.covariance.push_back(covariances[i](0,1));
        gaussian.covariance.push_back(covariances[i](1,0));
        gaussian.covariance.push_back(covariances[i](1,1));
        gmm_msg.gaussians.push_back(gaussian);
    }

    this->gmm_pub_->publish(gmm_msg);

    if ((GRAPHICS_ON) && (this->app_gui->isOpen()))
    {
        this->app_gui->clear();
        this->app_gui->drawGlobalReference(sf::Color(255,255,0), sf::Color(255,255,255));
        this->app_gui->drawPoint(pose);                         // draw robot
        this->app_gui->drawParticles(samples);                  // draw particles

        // Draw targets (only for visualization)
        for (int i = 0; i < TARGETS_NUM; i++)
        {
            this->app_gui->drawPoint(this->targets_real[i], sf::Color(0, 0, 255));          // draw real targets
        }

        //Display window
        this->app_gui->display();
    }
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Supervisor>());
    rclcpp::shutdown();
    return 0;
}