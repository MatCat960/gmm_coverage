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
#include "turtlebot3_msgs/msg/gaussian.hpp"
#include "turtlebot3_msgs/msg/gmm.hpp"
// #include <GaussianMixtureModel/GaussianMixtureModel.h>
// #include <GaussianMixtureModel/ExpectationMaximization.h>
// #include <GaussianMixtureModel/TrainSet.h>

#include "gaussian_mixture_model/gaussian_mixture_model.h"

#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;




class Interface : public rclcpp::Node
{

public:
    Interface() : Node("human_swarm_interface"), gmm(4)
    {
        // --------------------------------------------------------- ROS parameters ----------------------------------------------------------
        // Area parameters
        this->declare_parameter<double>("AREA_SIZE_x", 3.0);
        this->get_parameter("AREA_SIZE_x", AREA_SIZE_x);
        this->declare_parameter<double>("AREA_SIZE_y", 3.0);
        this->get_parameter("AREA_SIZE_y", AREA_SIZE_y);
        this->declare_parameter<double>("AREA_LEFT", -1.5);
        this->get_parameter("AREA_LEFT", AREA_LEFT);
        this->declare_parameter<double>("AREA_BOTTOM", -1.5);
        this->get_parameter("AREA_BOTTOM", AREA_BOTTOM);

        this->declare_parameter<int>("CLUSTERS_NUM", 4);
        this->get_parameter("CLUSTERS_NUM", CLUSTERS_NUM);
        this->declare_parameter<int>("PARTICLES_NUM", 200);
        this->get_parameter("PARTICLES_NUM", PARTICLES_NUM);
        
        // --------------------------------------------------------- GMM ROS publisher -------------------------------------------------------
        publisher = this->create_publisher<turtlebot3_msgs::msg::GMM>("/gaussian_mixture_model", 1);
        timer_ = this->create_wall_timer(100ms, std::bind(&Interface::timer_callback, this));

        //----------------------------------------------------------- init Variables ---------------------------------------------------------
        

        // ----------------------------------------------------------- SFML GUI --------------------------------------------------------------
        app_gui.reset(new Graphics{AREA_SIZE_x, AREA_SIZE_y, AREA_LEFT, AREA_BOTTOM, 2.0});
        
        
        drawPolygon();                                                                  // draw ROI and save vertices
        samples.resize(2, PARTICLES_NUM);
        samples = generateSamples(PARTICLES_NUM);                   // generate desired number of samples inside ROI              
        std::cout << "Samples generated with shape : " << samples.size() << "\n";
        gmm.fitgmm(samples, CLUSTERS_NUM, 1000, 1e-3, false);                  // fit GMM to samples
        std::cout << "Fitting completed...\n";
        mean_points = gmm.getMeans();
        covariances = gmm.getCovariances();
        weights = gmm.getWeights();
        std::cout << "GMM initialized\n";
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
    ~Interface()
    {
        if (this->app_gui->isOpen()){this->app_gui->close();}
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }


    void drawPolygon();
    // void generateSamples(Eigen::MatrixXd& samples_matrix, int n_samples);
    Eigen::MatrixXd generateSamples(int n_samples);
    bool insideROI(Eigen::VectorXd q, std::vector<Eigen::VectorXd> verts);


private:

    // ------------------------------ Area parameters ------------------------------------
    double AREA_SIZE_x;
    double AREA_SIZE_y;
    double AREA_LEFT;
    double AREA_BOTTOM;
    //-----------------------------------------------------------------------------------
    GaussianMixtureModel gmm;
    std::vector<Eigen::MatrixXd> covariances;
    std::vector<Eigen::VectorXd> mean_points;
    std::vector<double> weights;
    Eigen::MatrixXd samples;
    //------------------------- Publishers and subscribers ------------------------------
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<turtlebot3_msgs::msg::GMM>::SharedPtr publisher;
    turtlebot3_msgs::msg::GMM gmm_msg;
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
    int PARTICLES_NUM;


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

Eigen::MatrixXd Interface::generateSamples(int n_samples)
{
    std::vector<Eigen::VectorXd> samples;

    // Get min and max values of x and y
    double x_min, y_min = 100;
    double x_max, y_max = -100;
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

    return displayMatrix;
}




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Interface>());
    rclcpp::shutdown();
    return 0;
}
