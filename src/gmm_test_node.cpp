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


#include "rclcpp/rclcpp.hpp"
#include "turtlebot3_msgs/msg/gaussian.hpp"
#include "turtlebot3_msgs/msg/gmm.hpp"


#include "gaussian_mixture_model/gaussian_mixture_model.h"



class GMM_Test : public rclcpp::Node
{
private:
    GaussianMixtureModel gmm;
    std::vector<Eigen::MatrixXd> covariances;
    std::vector<Eigen::VectorXd> mean_points;
    std::vector<double> weights;
    Eigen::MatrixXd samples;

    // ROS Related Variables
    rclcpp::TimerBase::SharedPtr timer_;
    turtlebot3_msgs::msg::GMM gmm_msg;
    rclcpp::Publisher<turtlebot3_msgs::msg::GMM>::SharedPtr gmm_pub_;

public:

    // initialize GMM with 5 random components
    GMM_Test(): Node("gmm_test"), gmm(4)
    {
        std::cout << "Constructor called" << std::endl;

        // Initialize ROS variables
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&GMM_Test::timerCallback, this));
        gmm_pub_ = this->create_publisher<turtlebot3_msgs::msg::GMM>("/gaussian_mixture_model", 10);

        samples.resize(2,200);
        double dev = 0.5;
        std::default_random_engine gen;

        // Set desired values
        Eigen::VectorXd p1(2);
        p1 << -6.0, 0.0;
        // Generate samples
        std::normal_distribution<double> dist_x(p1(0), dev);
        std::normal_distribution<double> dist_y(p1(1), dev);

        for(int i = 0; i < 50; i++)
        {
            samples(0,i) = dist_x(gen);
            samples(1,i) = dist_y(gen);
        }

        // Set desired values
        Eigen::VectorXd p2(2);
        p2 << -4.0, 4.0;
        // Generate samples
        std::normal_distribution<double> dist_x2(p2(0), dev);
        std::normal_distribution<double> dist_y2(p2(1), dev);

        for(int i = 50; i < 100; i++)
        {
            samples(0,i) = dist_x2(gen);
            samples(1,i) = dist_y2(gen);
        }


        // Set desired values
        Eigen::VectorXd p3(2);
        p3 << 4.0, 6.0;
        // Generate samples
        std::normal_distribution<double> dist_x3(p3(0), dev);
        std::normal_distribution<double> dist_y3(p3(1), dev);

        for(int i = 100; i < 150; i++)
        {
            samples(0,i) = dist_x3(gen);
            samples(1,i) = dist_y3(gen);
        }

        // Set desired values
        Eigen::VectorXd p4(2);
        p4 << 4.0, -6.0;
        // Generate samples
        std::normal_distribution<double> dist_x4(p4(0), dev);
        std::normal_distribution<double> dist_y4(p4(1), dev);

        for(int i = 150; i < 200; i++)
        {
            samples(0,i) = dist_x4(gen);
            samples(1,i) = dist_y4(gen);
        }

        auto timerstart = std::chrono::high_resolution_clock::now();
        gmm.fitgmm(samples, 4, 1000, 1e-3, false);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout<<"Computation time for EM: -------------: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - timerstart).count()<<" ms :-------------\n";

        mean_points = gmm.getMeans();
        covariances = gmm.getCovariances();
        weights = gmm.getWeights();

        std::cout << "GMM Initialized" << std::endl;

        std::cout << "Mean Points: " << std::endl;
        for(int i = 0; i < mean_points.size(); i++)
        {
            std::cout << mean_points[i] << std::endl;
        }

        std::cout << "Covariances: " << std::endl;
        for(int i = 0; i < covariances.size(); i++)
        {
            std::cout << covariances[i] << std::endl;
        }

        std::cout << "Weights: " << std::endl;
        for(int i = 0; i < weights.size(); i++)
        {
            std::cout << weights[i] << std::endl;
        }

        // Create ROS msg
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

        std::cout << "ROS MSG Initialized" << std::endl;

    }

    ~GMM_Test()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    void timerCallback()
    {        
        gmm_pub_->publish(gmm_msg);
    }

    

};//End of class SubscribeAndPublish


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GMM_Test>());
    rclcpp::shutdown();
    return 0;
}