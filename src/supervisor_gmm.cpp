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
#include "gmm_msgs/msg/gaussian.hpp"
#include "gmm_msgs/msg/gmm.hpp"

#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;




class Supervisor : public rclcpp::Node
{

public:
    Supervisor() : Node("gmm_supervisor")
    {
        // MODE: 0 = Default, 1 = FROM FILE
        this->declare_parameter<int>("MODE", 1);
        this->get_parameter("MODE", MODE);
                    int column = 1;

        this->declare_parameter<std::string>("FILE_PATH", "/home/mattia/arscontrol_turtlebot/src/gmm_coverage/gmm_matrix.txt");
        this->get_parameter("FILE_PATH", FILE_PATH);

        // --------------------------------------------------------- GMM ROS publisher -------------------------------------------------------
        publisher = this->create_publisher<gmm_msgs::msg::GMM>("/gaussian_mixture_model", 1);
        timer_ = this->create_wall_timer(100ms, std::bind(&Supervisor::timer_callback, this));

        //----------------------------------------------------------- init Variables ---------------------------------------------------------
        if (MODE == 0)
        {
            // VARs = {{{1.86,-0.02},{-0.02,19.26}},{{8.02,-0.27},{-0.27,2.03}},{{1.87,0.38},{0.38,1.9}},{{2.22,0.19},{0.19,5.58}}};
            // MEANs = {{-3.8337,0},{5.5234,-3.0563},{-2.8070,3.0},{-2.8554,-2.9132}};
            // weights = {0.4183,0.2817,0.1105,0.1895};
            // MEANs = {{-0.3,0.0}};
            // VARs = {{{0.2, 0.0},{0.0,0.5}}};
            // weights = {1};
            // VARs = {{{1.86,-0.02},{-0.02,19.26}},{{8.02,-0.27},{-0.27,2.03}}};
            // MEANs = {{-0.8337,0},{2.5234,-3.0563}};
            // weights = {0.41,0.59};

            // ------ test gmm ------
            VARs = {{{0.098,0.002},{0.002,0.0167}},{{0.014,-0.002},{-0.002,0.008}},{{0.017,-0.003},{-0.003,0.03}},{{0.02,-0.007},{-0.007,0.11}}};
            MEANs = {{0.47,-1.07},{-0.21,0.34},{-0.19,-1.01},{-0.17,-0.31}};
            weights = {0.4174,0.0742,0.1430,0.3654};

        } else if (MODE == 1)
        {
            std::fstream file;
            std::string delimiter = "\t";
            file.open(FILE_PATH, std::ios::in);
            if (file.is_open())
            {
                std::string line;
                std::string token;
                float w;
                std::vector<std::vector<float>> covar;
                while (getline(file, line))
                {
                    size_t pos = 0;
                    std::vector<float> mean;
                    std::vector<float> v;
                    
                    while ((pos = line.find(delimiter)) != std::string::npos)
                    {
                        token = line.substr(0,pos);
                        line.erase(0, pos + delimiter.length());
                        v.push_back(std::stof(token));
                    }
                        v.push_back(std::stof(line));
                        // se il primo elem = 0 sono nella seconda riga, altrimenti nella prima
                        std::vector<float> row;
                        if (v[0] != 0)
                        {
                            mean.push_back(v[0]);
                            mean.push_back(-v[1]);
                            row.push_back(v[2]);
                            row.push_back(v[3]);
                            MEANs.push_back(mean);
                            covar.push_back(row);
                            weights.push_back(v[4]);
                        } else
                        {
                            row.push_back(v[2]);
                            row.push_back(v[3]);
                            covar.push_back(row);
                            VARs.push_back(covar);
                            covar.clear();
                        }
                
                }
                file.close();
            }
        }
        
        //------------------------------------------------------------------------------------------------------------------------------------

        // for (int i=0; i<MEANs.size(); i++)
        // {
        //     std::cout << "Media: " << std::to_string(MEANs[i][0]) << ", " << std::to_string(MEANs[i][1]) << std::endl;
        //     std::cout << "Matrice covarianza: \n";
        //     for (int j=0; j<VARs[0].size(); j++)
        //     {
        //         std::cout << std::to_string(VARs[i][j][0]) << ", " << std::to_string(VARs[i][j][1]) << std::endl;
        //     }
        //     std::cout << "Peso: " << std::to_string(weights[i]) << std::endl;
        // }

        
        // ------------------------------------------ Creazione messaggi custom Gaussiane --------------------------------------------
        // gmm_msgs::msg::GMM gmm_msg;
        for (int i=0; i<MEANs.size(); i++)
        {
            gmm_msgs::msg::Gaussian gaussian_msg;
            geometry_msgs::msg::Point mean_pt;

            mean_pt.x = MEANs[i][0];
            mean_pt.y = MEANs[i][1];
            mean_pt.z = 0;
            gaussian_msg.mean_point = mean_pt;
            for (int j=0; j<VARs[0].size(); j++)
            {
                gaussian_msg.covariance.push_back(VARs[i][j][0]);
                gaussian_msg.covariance.push_back(VARs[i][j][1]);
            }
            this->gmm_msg.gaussians.push_back(gaussian_msg);
            this->gmm_msg.weights.push_back(weights[i]);
        }
    }
    ~Supervisor()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }



private:

    int MODE;
    std::string FILE_PATH;
    std::vector<std::vector<float>> MEANs;
    std::vector<std::vector<std::vector<float>>> VARs;
    std::vector<float> weights;

    //------------------------- Publishers and subscribers ------------------------------
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<gmm_msgs::msg::GMM>::SharedPtr publisher;
    gmm_msgs::msg::GMM gmm_msg;
    //-----------------------------------------------------------------------------------


    //timer - check how long robots are being stopped
    time_t timer_init_count;
    time_t timer_final_count;

    void timer_callback()
    {
        // RCLCPP_INFO(this->get_logger(), "Sto pubblicando il messaggio");
        publisher->publish(this->gmm_msg);
    }
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Supervisor>());
    rclcpp::shutdown();
    return 0;
}
