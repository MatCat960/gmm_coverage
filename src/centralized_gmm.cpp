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

//Robots parameters ------------------------------------------------------
const double MAX_ANG_VEL = 0.3;
const double MAX_LIN_VEL = 0.2;         //set to turtlebot max velocities
const double b = 0.025;                 //for differential drive control (only if we are moving a differential drive robot (e.g. turtlebot))
//------------------------------------------------------------------------
const bool centralized_centroids = false;   //compute centroids using centralized computed voronoi diagram
const float CONVERGENCE_TOLERANCE = 0.1;
//------------------------------------------------------------------------
const int shutdown_timer = 15;           //count how many seconds to let the robots stopped before shutting down the node

// std::vector<std::vector<double>> corners;

// Corners
// std::vector<Vector2<double>> corners{{-10.0,-10.0},{10.0,-10.0},{-10.0,-7.0},{-10.0,-5.0},{10.0,-3.0},{10.0,-1.0},{-10.0,1.0},{-10.0,3.0},{10.0,5.0},{10.0,7.0},{-10.0,10.0},{10.0,10.0}};

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

class Controller : public rclcpp::Node
{

public:
    Controller() : Node("gmm_distribution")
    {
        //------------------------------------------------- ROS parameters ---------------------------------------------------------
        this->declare_parameter<int>("ROBOTS_NUM", 12);
        this->get_parameter("ROBOTS_NUM", ROBOTS_NUM);
        this->declare_parameter<bool>("SIM",true);
        this->get_parameter("SIM", SIM);

        //Range di percezione singolo robot (= metà lato box locale)
        this->declare_parameter<double>("ROBOT_RANGE", 4);
        this->get_parameter("ROBOT_RANGE", ROBOT_RANGE);
        
        // Parameters for Gaussian
        this->declare_parameter<bool>("GAUSSIAN_DISTRIBUTION", 1);
        this->get_parameter("GAUSSIAN_DISTRIBUTION", GAUSSIAN_DISTRIBUTION);
        this->declare_parameter<double>("PT_X", -100);
        this->get_parameter("PT_X", PT_X);
        this->declare_parameter<double>("PT_Y", 100);
        this->get_parameter("PT_Y", PT_Y);
        this->declare_parameter<double>("VAR", 2);
        this->get_parameter("VAR", VAR);

        //view graphical voronoi rapresentation - bool
        this->declare_parameter<bool>("GRAPHICS_ON", true);
        this->get_parameter("GRAPHICS_ON", GRAPHICS_ON);

        // Area parameter
        this->declare_parameter<double>("AREA_SIZE_x", 40);
        this->get_parameter("AREA_SIZE_x", AREA_SIZE_x);
        this->declare_parameter<double>("AREA_SIZE_y", 20);
        this->get_parameter("AREA_SIZE_y", AREA_SIZE_y);
        this->declare_parameter<double>("AREA_LEFT", -20);
        this->get_parameter("AREA_LEFT", AREA_LEFT);
        this->declare_parameter<double>("AREA_BOTTOM", -10);
        this->get_parameter("AREA_BOTTOM", AREA_BOTTOM);
        //-----------------------------------------------------------------------------------------------------------------------------------

        //--------------------------------------------------- Subscribers and Publishers ----------------------------------------------------
        if (SIM)
        {
            for (int i = 0; i < ROBOTS_NUM; i++)
            {
                odomSub_.push_back(this->create_subscription<nav_msgs::msg::Odometry>("/turtlebot" + std::to_string(i) + "/odom", 100, [this, i](nav_msgs::msg::Odometry::SharedPtr msg) {this->odomCallback(msg,i);}));
                velPub_.push_back(this->create_publisher<geometry_msgs::msg::Twist>("/turtlebot" + std::to_string(i) + "/cmd_vel", 1));
            }
        } else
        {
            for (int i = 0; i < ROBOTS_NUM; i++)
            {
                velPub_.push_back(this->create_publisher<geometry_msgs::msg::Twist>("/turtle" + std::to_string(i) + "/cmd_vel", 1));
                poseSub_.push_back(this->create_subscription<geometry_msgs::msg::PoseStamped>("/vrpn_client_node/turtle" + std::to_string(i) + "/pose", 100, [this, i](geometry_msgs::msg::PoseStamped::SharedPtr msg) {this->poseCallback(msg,i);}));        
            }
        }
        joySub_ = this->create_subscription<geometry_msgs::msg::Twist>("/joy_vel", 1, std::bind(&Controller::joy_callback, this, _1));
        gmmSub_ = this->create_subscription<gmm_msgs::msg::GMM>("/gaussian_mixture_model", 1, std::bind(&Controller::gmm_callback, this, _1));
        timer_ = this->create_wall_timer(500ms, std::bind(&Controller::Formation, this));
        //rclcpp::on_shutdown(std::bind(&Controller::stop,this));

        //----------------------------------------------------------- init Variables ---------------------------------------------------------
        pose_x = Eigen::VectorXd::Zero(ROBOTS_NUM);
        pose_y = Eigen::VectorXd::Zero(ROBOTS_NUM);
        pose_theta = Eigen::VectorXd::Zero(ROBOTS_NUM);
        time(&this->timer_init_count);
        time(&this->timer_final_count);
        got_gmm = false;
        
        //----------------------------------------------- Graphics window -----------------------------------------------
        if (GRAPHICS_ON)
        {
            app_gui.reset(new Graphics{AREA_SIZE_x, AREA_SIZE_y, AREA_LEFT, AREA_BOTTOM, VAR});
        }
        //---------------------------------------------------------------------------------------------------------------
    }
    ~Controller()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    //void stop(int signum);
    void stop();
    void test_print();
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int j);
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, int j);
    void joy_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void gmm_callback(const gmm_msgs::msg::GMM::SharedPtr msg);
    Eigen::VectorXd Matrix_row_sum(Eigen::MatrixXd x);
    Eigen::MatrixXd Diag_Matrix(Eigen::VectorXd V);
    void Formation();
    geometry_msgs::msg::Twist Diff_drive_compute_vel(double vel_x, double vel_y, double alfa);


private:
    int ROBOTS_NUM;
    double ROBOT_RANGE;
    bool NotJustStarted;           // Flag per indicare che sono al primo ciclo di esecuzione del nodo
    bool got_gmm;
    bool SIM;

    double vel_linear_x, vel_angular_z;
    Eigen::VectorXd pose_x;
    Eigen::VectorXd pose_y;
    Eigen::VectorXd pose_theta;
    std::vector<Vector2<double>> seeds_xy;
    int seeds_counter = 0;
    std::vector<std::vector<float>> MEANs;
    std::vector<std::vector<std::vector<float>>> VARs;
    std::vector<float> weights;
    std::vector<std::vector<double>> corners;
    std::vector<double> position;
    std::vector<std::vector<Vector2<double>>> old_positions;                    // vector containing last 10 positions of each robot

    //------------------------- Publishers and subscribers ------------------------------
    std::vector<rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr> velPub_;
    // std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> poseSub_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> poseSub_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> odomSub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr joySub_;
    rclcpp::Subscription<gmm_msgs::msg::GMM>::SharedPtr gmmSub_;
    rclcpp::TimerBase::SharedPtr timer_;
    gmm_msgs::msg::GMM gmm_msg;
    //-----------------------------------------------------------------------------------

    //Rendering with SFML
    //------------------------------ graphics window -------------------------------------
    std::unique_ptr<Graphics> app_gui;
    //------------------------------------------------------------------------------------

    //---------------------------- Environment definition --------------------------------
    double AREA_SIZE_x;
    double AREA_SIZE_y;
    double AREA_LEFT;
    double AREA_BOTTOM;
    //------------------------------------------------------------------------------------

    //---------------------- Gaussian Density Function parameters ------------------------
    bool GAUSSIAN_DISTRIBUTION;
    double PT_X;
    double PT_Y;
    double VAR;

    //------------------------------------------------------------------------------------

    //graphical view - ON/OFF
    bool GRAPHICS_ON;

    //timer - check how long robots are being stopped
    time_t timer_init_count;
    time_t timer_final_count;

    //ofstream on external log file
    // std::ofstream log_file;
    // std::ofstream gauss_file;
    // std::ofstream k_file;
    long unsigned int log_line_counter=0;
};



void Controller::test_print()
{
    std::cout<<"ENTERED"<<std::endl;
}

void Controller::stop()
{
    //if (signum == SIGINT || signum == SIGKILL || signum ==  SIGQUIT || signum == SIGTERM)
    RCLCPP_INFO_STREAM(this->get_logger(), "shutting down the controller, stopping the robots, closing the graphics window");
    if ((GRAPHICS_ON) && (this->app_gui->isOpen())){
        this->app_gui->close();
    }
    this->timer_->cancel();
    rclcpp::sleep_for(100000000ns);

    geometry_msgs::msg::Twist vel_msg;
    for (int i = 0; i < 100; ++i)
    {
        for (int r = 0; r < ROBOTS_NUM; ++r)
        {
            this->velPub_[r]->publish(vel_msg);
        }
    }

    RCLCPP_INFO_STREAM(this->get_logger(), "controller has been closed and robots have been stopped");
    rclcpp::sleep_for(100000000ns);
    // this->close_log_file();
    // this->close_gauss_file();
    // this->close_k_file();
}

void Controller::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, int j)
{
    // std::cout << "ROBOT " << std::to_string(j) << " dentro a poseCallback" << std::endl;
    this->pose_x(j) = msg->pose.position.x;
    this->pose_y(j) = msg->pose.position.y;

    tf2::Quaternion q(
    msg->pose.orientation.x,
    msg->pose.orientation.y,
    msg->pose.orientation.z,
    msg->pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    this->pose_theta(j) = yaw;

    // std::cout << "ROBOT " << std::to_string(j) << " pose_x: " << std::to_string(this->pose_x(j)) << ", pose_y: " << std::to_string(this->pose_y(j)) << ", theta: " << std::to_string(this->pose_theta(j)) << std::endl;
}

void Controller::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int j)
{
    // std::cout << "ROBOT " << std::to_string(ID) << " dentro a odomCallback" << std::endl;
    this->pose_x(j) = msg->pose.pose.position.x;
    this->pose_y(j) = msg->pose.pose.position.y;

    tf2::Quaternion q(
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z,
    msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    this->pose_theta(j) = yaw;
}

void Controller::joy_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    this->vel_linear_x = msg->linear.x;
    this->vel_angular_z = msg->angular.z;
}

void Controller::gmm_callback(const gmm_msgs::msg::GMM::SharedPtr msg)
{
    // std::cout << "ROBOT " << std::to_string(ID) << " dentro a GMM callback" << std::endl;
    this->gmm_msg.gaussians = msg->gaussians;
    this->gmm_msg.weights = msg->weights;
    this->got_gmm = true;
    // RCLCPP_INFO_STREAM(this->get_logger(), "Sto ricevendo il GMM");
}

Eigen::VectorXd Controller::Matrix_row_sum(Eigen::MatrixXd X)
{
    Eigen::VectorXd vect(X.rows());
    auto temp = X.rowwise().sum();
    vect << temp;
    
    return vect;
}

Eigen::MatrixXd Controller::Diag_Matrix(Eigen::VectorXd V)
{
    Eigen::MatrixXd M(V.size(), V.size());
    for (int i = 0; i < V.size(); ++i)
    {
        for (int j = 0; j < V.size(); ++j)
        {
            if (i == j)
            {
                M(i,j) = V(i);
            }
            else{

                M(i,j) = 0;
            }
        }
    }
    return M;
}


geometry_msgs::msg::Twist Controller::Diff_drive_compute_vel(double vel_x, double vel_y, double alfa){
    //-------------------------------------------------------------------------------------------------------
    //Compute velocities commands for the robot: differential drive control, for UAVs this is not necessary
    //-------------------------------------------------------------------------------------------------------

    geometry_msgs::msg::Twist vel_msg;
    //double alfa = (this->pose_theta(i));
    double v=0, w=0;

    v = cos(alfa) * vel_x + sin(alfa) * vel_y;
    w = -(1 / b) * sin(alfa) * vel_x + (1 / b) * cos(alfa) * vel_y;

    if (abs(v) <= MAX_LIN_VEL)
    {
        vel_msg.linear.x = v;
    }
    else {
        if (v >= 0)
        {
            vel_msg.linear.x = MAX_LIN_VEL;
        } else {
            vel_msg.linear.x = -MAX_LIN_VEL;
        }
    }

    if (abs(w) <= MAX_ANG_VEL)
    {
        vel_msg.angular.z = w;
    }
    else{
        if (w >= 0)
        {
            vel_msg.angular.z = MAX_ANG_VEL;
        } else {
            vel_msg.angular.z = -MAX_ANG_VEL;
        }
    }
    return vel_msg;
}


void Controller::Formation()
{
    if(!this->got_gmm) return;	
    // std::cout << "GMM weights: " << this->gmm_msg.weights[10] << std::endl;
    // std::cout << "INIZIO ESPLORAZIONE" << std::endl;
    auto start = this->get_clock()->now().nanoseconds();
    //Parameters
    //double min_dist = 0.4;         //avoid robot collision
    int K_gain = 1;                  //Lloyd law gain
    this->log_line_counter = this->log_line_counter + 1;

    //Variables
    double vel_x=0.0, vel_y=0.0, vel_z = 0.0;
    std::vector<Vector2<double>> seeds;
    std::vector<std::vector<float>> centroids;
    std::vector<double> vel; std::vector<float> centroid;


    // ------------------------------------------------------ Environment definition -----------------------------------------------------
    Box<double> AreaBox{AREA_LEFT, AREA_BOTTOM, AREA_SIZE_x + AREA_LEFT, AREA_SIZE_y + AREA_BOTTOM};
    Box<double> RangeBox{-ROBOT_RANGE, -ROBOT_RANGE, ROBOT_RANGE, ROBOT_RANGE};

    // --------------- 18 x 15 m ------------------------
    Box<double> WallBox1{-7,-3.5,-2,3.5};
    Box<double> WallBox2{-6.5,-1.5,1.5,-1.5};
    std::vector<Box<double>> ObstacleBoxes = {};

    // std::cout << "Number of robots: " << ROBOTS_NUM << std::endl;

    // for (int i=0; i<ROBOTS_NUM; i++)
    // {
    //     // std::cout << "POSIZIONE ROBOT " << std::to_string(i) << ": " << std::to_string(this->pose_x(i)) << " " << std::to_string(this->pose_y(i)) << std::endl;
    //     while ((this->pose_x(i) == 0.0) && (this->pose_y(i) == 0.0))
    //     {
    //         // std::cout << "Waiting for robot " << std::to_string(i) << " to be localized..." << std::endl;
    //     }
    // }
    
    for (int i = 0; i < ROBOTS_NUM; ++i)
    {
        if ((this->pose_x(i) != 0.0) && (this->pose_y(i) != 0.0))
        {
            seeds.push_back({this->pose_x(i), this->pose_y(i)});    
        }
        // centroids.push_back({this->pose_x(i), this->pose_y(i)});
    }


    bool all_robots_stopped = true;

    for (int i = 0; i < ROBOTS_NUM; ++i)
    {
        // std::cout << "Number of obstacles: " << ObstacleBoxes.size() << std::endl;
        if (!centralized_centroids)     //flag x distributed computation
        {
            if (seeds.size() >= 1)
            {
                //-----------------Voronoi--------------------
                //Rielaborazione vettore "points" globale in coordinate locali
                auto local_seeds_i = reworkPointsVector(seeds, seeds[i]);
                
                //Filtraggio siti esterni alla box (simula azione del sensore)
                auto flt_seeds = filterPointsVector(local_seeds_i, RangeBox);
                auto diagram = generateDecentralizedDiagram(flt_seeds, RangeBox, seeds[i], ROBOT_RANGE, AreaBox);

                //compute centroid -- GAUSSIAN DISTRIBUTION
                centroid = computeGMMPolygonCentroid2(diagram, this->gmm_msg, ObstacleBoxes);
                // std::cout << "Centroid: x = " << centroid[0] << ", y = " << centroid[1] <<std::endl;
            }
        }

        double norm = sqrt(centroid[0]*centroid[0] + centroid[1]*centroid[1]);
        if (norm > CONVERGENCE_TOLERANCE)
        {
            vel_x = K_gain*(centroid[0]);
            vel_y = K_gain*(centroid[1]);
            vel_z = K_gain*(centroid[2]);
            all_robots_stopped = false;
        } else {
            vel_x = 0.0;
            vel_y = 0.0;
            vel_z = 0.0;
            std::cout << "ROBOT " << i << ": STOPPED" << std::endl;
        }

        std::cout<<"sending velocities to " << i << ":: " << vel_x << ", "<<vel_y<<std::endl;

        //-------------------------------------------------------------------------------------------------------
        //Compute velocities commands for the robot: differential drive control, for UAVs this is not necessary
        //-------------------------------------------------------------------------------------------------------
        auto vel_msg = this->Diff_drive_compute_vel(vel_x, vel_y, this->pose_theta(i));
        //-------------------------------------------------------------------------------------------------------

        RCLCPP_INFO_STREAM(get_logger(), "sending cmd_vel to " << i << ":: " << vel_msg.angular.z << ", "<<vel_msg.linear.x);

        this->velPub_[i]->publish(vel_msg);
        //-------------------------------------------------------------------------------------------------------
        // if (!centralized_centroids)
        // {
        // this->write_log_file(std::to_string(i) + "\t\t" + std::to_string(this->pose_x(i)) + "\t\t" + std::to_string(this->pose_y(i)) + "\t");
        // }
    }


if ((GRAPHICS_ON) && (this->app_gui->isOpen())){
    // check_window_event();

    if (seeds.size() >= 2)
    {
        this->app_gui->clear();
        auto diagram = generateCentralizedDiagram(seeds, AreaBox);
        // if (centralized_centroids){                         //flag x centralized centroids computation
        //     centroids = diagram.computeLloydRelaxation();
        // }
        this->app_gui->drawDiagram(diagram);
        this->app_gui->drawPoints(diagram);
        this->app_gui->drawGlobalReference(sf::Color(255,255,0), sf::Color(255,255,255));
        // this->app_gui->drawGaussianContours(MEANs, VARs);
        this->app_gui->drawGMM(this->gmm_msg);
        for (int i=0; i<ObstacleBoxes.size(); ++i)
        {
            this->app_gui->drawObstacle(ObstacleBoxes[i], sf::Color(255,255,255));
        }
        //Display window
        this->app_gui->display();
    }
}

if (all_robots_stopped == true)
    {
        time(&this->timer_final_count);
        if (this->timer_final_count - this->timer_init_count >= shutdown_timer)
        {
            //shutdown node
            std::cout<<"SHUTTING DOWN THE NODE"<<std::endl;
            this->stop();   //stop the controller
            rclcpp::shutdown();
        }
    } else {
        time(&this->timer_init_count);
    }

auto end = this->get_clock()->now().nanoseconds();
std::cout<<"Computation time cost: -----------------: "<<end - start<<std::endl;

}


//alternatively to a global variable to have access to the method you can make STATIC the class method interested, 
//but some class function may not be accessed: "this->" method cannot be used

std::shared_ptr<Controller> globalobj_signal_handler;     //the signal function requires only one argument {int}, so the class and its methods has to be global to be used inside the signal function.
void nodeobj_wrapper_function(int){
    std::cout<<"signal handler function CALLED"<<std::endl;
    globalobj_signal_handler->stop();
}

int main(int argc, char **argv)
{
    signal(SIGINT, nodeobj_wrapper_function);

    rclcpp::init(argc, argv);
    auto node = std::make_shared<Controller>();

    globalobj_signal_handler = node;    //to use the ros function publisher, ecc the global pointer has to point to the same node object.

    rclcpp::spin(node);

    rclcpp::sleep_for(100000000ns);
    rclcpp::shutdown();

    return 0;
}
