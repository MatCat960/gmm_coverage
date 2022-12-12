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
#include "gmm_coverage/Graphics.h"
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
#include "gmm_msgs/msg/gaussian.hpp"
#include "gmm_msgs/msg/gmm.hpp"
#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include <GaussianMixtureModel/TrainSet.h>

#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;




class Interface : public rclcpp::Node
{

public:
    Interface() : Node("human_swarm_interface")
    {
        // --------------------------------------------------------- ROS parameters ----------------------------------------------------------
        // Area parameters
        this->declare_parameter<double>("AREA_SIZE_x", 10);
        this->get_parameter("AREA_SIZE_x", AREA_SIZE_x);
        this->declare_parameter<double>("AREA_SIZE_y", 10);
        this->get_parameter("AREA_SIZE_y", AREA_SIZE_y);
        this->declare_parameter<double>("AREA_LEFT", -5);
        this->get_parameter("AREA_LEFT", AREA_LEFT);
        this->declare_parameter<double>("AREA_BOTTOM", -5);
        this->get_parameter("AREA_BOTTOM", AREA_BOTTOM);

        this->declare_parameter<int>("CLUSTERS_NUM", 4);
        this->get_parameter("CLUSTERS_NUM", CLUSTERS_NUM);
        
        // --------------------------------------------------------- GMM ROS publisher -------------------------------------------------------
        publisher = this->create_publisher<gmm_msgs::msg::GMM>("/gaussian_mixture_model", 1);
        timer_ = this->create_wall_timer(100ms, std::bind(&Interface::timer_callback, this));

        //----------------------------------------------------------- init Variables ---------------------------------------------------------
        

        // ----------------------------------------------------------- SFML GUI --------------------------------------------------------------
        app_gui.reset(new Graphics{AREA_SIZE_x, AREA_SIZE_y, AREA_LEFT, AREA_BOTTOM, 2.0});
        
        
        drawPolygon();                                                                  // draw ROI and save vertices
        std::vector<Eigen::VectorXd> samples = generateSamples(2000);                   // generate desired number of samples inside ROI              
        gauss::TrainSet samples_set(samples);                                           // create train set from samples
        std::vector<gauss::gmm::Cluster> clusters = gauss::gmm::ExpectationMaximization(samples_set, CLUSTERS_NUM); // run EM algorithm to get GMM
        gauss::gmm::GaussianMixtureModel gmm_(clusters);                               // create GMM from clusters
        std::cout << "GMM initialized\n";
        for (int i=0; i<gmm_.getClusters().size(); i++)
        {
            std::cout << "Cluster " << i << ": weight = " << gmm_.getClusters()[i].weight << std::endl;
            std::cout << "Mean: " << gmm_.getClusters()[i].distribution->getMean().transpose() << std::endl;
            std::cout << "Covariance matrix: \n" << gmm_.getClusters()[i].distribution->getCovariance().transpose() << std::endl;
        }

        // Create GMM ROS msg
        for (int i=0; i<gmm_.getClusters().size(); i++)
        {
            gmm_msgs::msg::Gaussian gaussian_msg;
            geometry_msgs::msg::Point mean_pt;

            mean_pt.x = gmm_.getClusters()[i].distribution->getMean()[0];
            mean_pt.y = gmm_.getClusters()[i].distribution->getMean()[1];
            mean_pt.z = 0.0;
            gaussian_msg.mean_point = mean_pt;
            for (int j=0; j<gmm_.getClusters()[i].distribution->getCovariance().rows(); j++)
            {
                gaussian_msg.covariance.push_back(gmm_.getClusters()[i].distribution->getCovariance()(j,0));
                gaussian_msg.covariance.push_back(gmm_.getClusters()[i].distribution->getCovariance()(j,1));
            }

            gmm_msg.gaussians.push_back(gaussian_msg);
            gmm_msg.weights.push_back(gmm_.getClusters()[i].weight);
        }

    }
    ~Interface()
    {
        if (this->app_gui->isOpen()){this->app_gui->close();}
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }


    void drawPolygon();
    std::vector<Eigen::VectorXd> generateSamples(int n_samples);
    bool insideROI(Eigen::VectorXd q, std::vector<Eigen::VectorXd> verts);


private:

    // ------------------------------ Area parameters ------------------------------------
    double AREA_SIZE_x;
    double AREA_SIZE_y;
    double AREA_LEFT;
    double AREA_BOTTOM;
    //-----------------------------------------------------------------------------------

    //------------------------- Publishers and subscribers ------------------------------
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<gmm_msgs::msg::GMM>::SharedPtr publisher;
    gmm_msgs::msg::GMM gmm_msg;
    //-----------------------------------------------------------------------------------

    // SFML GUI
    //------------------------------ graphics window -------------------------------------
    std::unique_ptr<Graphics> app_gui;
    //------------------------------------------------------------------------------------

    //timer - check how long robots are being stopped
    time_t timer_init_count;
    time_t timer_final_count;

    std::vector<Eigen::VectorXd> vertices;
    int CLUSTERS_NUM;


    void timer_callback()
    {
        publisher->publish(this->gmm_msg);
    }
};


void Interface::drawPolygon()
{
    if (this->app_gui->isOpen())
    {
        bool poly_complete = false;
        this->app_gui->clear();

        Eigen::MatrixXd displayMatrix(2,0);
        
        while (!poly_complete)
        {
            Eigen::VectorXd p(2);
            this->app_gui->drawGlobalReference(sf::Color(255,255,0), sf::Color(255,255,255));
            auto v = this->app_gui->drawROI();                          // v : std::pair<Eigen::VectorXd, bool>
            p = v.first;
            poly_complete = v.second;
            if (p(0) != 0 && p(1) != 0)
            {
                this->vertices.push_back(p);
                std::cout << "p: " << p.transpose() << std::endl;
                displayMatrix.conservativeResize(displayMatrix.rows(), displayMatrix.cols()+1);
                displayMatrix.col(displayMatrix.cols()-1) = p;
                
            }

            // Draw points and lines (lines only if at least 2 points)
            this->app_gui->drawParticles(displayMatrix);
            if (this->vertices.size() > 1)
            {
                for (int i=0; i<this->vertices.size()-1; i++)
                {
                    Vector2<double> origin = {this->vertices[i](0),this->vertices[i](1)};
                    Vector2<double> destination = {this->vertices[i+1](0),this->vertices[i+1](1)};
                    this->app_gui->drawEdge_global(origin, destination, sf::Color(255,255,0));
                }
            }
            this->app_gui->display();

        }

    }
}


bool Interface::insideROI(Eigen::VectorXd q, std::vector<Eigen::VectorXd> verts)
{
    // Draw a horizontal line from the point to the right, and count the number of times it intersects with the polygon.
    // If the number is odd, the point is inside the polygon.
    int i,j = 0;
    int nvert = verts.size();
    bool c = false;

    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((verts[i](1)>q(1)) != (verts[j](1)>q(1))) &&
            (q(0) < (verts[j](0)-verts[i](0)) * (q(1)-verts[i](1)) / (verts[j](1)-verts[i](1)) + verts[i](0)) )
            c = !c;
    }

    return c;
}

std::vector<Eigen::VectorXd> Interface::generateSamples(int n_samples)
{
    std::vector<Eigen::VectorXd> samples;

    // Get min and max values of x and y
    double x_min, x_max, y_min, y_max = -100;
    for (int i=0; i<this->vertices.size(); i++)
    {
        double x = this->vertices[i](0);
        double y = this->vertices[i](1);

        if (x < x_min){x_min = x;}
        if (x > x_max){x_max = x;}
        if (y < y_min){y_min = y;}
        if (y > y_max){y_max = y;}
    }

    // Generate samples in given area
    std::default_random_engine gen;
    std::uniform_real_distribution<> distr_x(x_min, x_max);
    std::uniform_real_distribution<> distr_y(y_min, y_max);
    int count = 0;

    while (count < n_samples)
    {
        double x = distr_x(gen);
        double y = distr_y(gen);
        Eigen::VectorXd p(2);
        p << x, y;
        if (insideROI(p, this->vertices))
        {
            samples.push_back(p);
            count++;
        }
    }

    // Convert samples to Eigen matrix and display
    Eigen::MatrixXd displayMatrix(2,n_samples);
    for (int i=0; i<n_samples; i++)
    {
        displayMatrix.col(i) = samples[i];
    }
    this->app_gui->drawParticles(displayMatrix);
    this->app_gui->display();

    return samples;
}




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Interface>());
    rclcpp::shutdown();
    return 0;
}
