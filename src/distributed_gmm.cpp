// STL
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>
#include <tf2/utils.h>
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
#include "gmm_coverage/Diagram.h"
// #include "Graphics.h"
// ROS includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "sensor_msgs/msg/channel_float32.hpp"
#include "std_msgs/msg/int16.hpp"
#include "std_msgs/msg/bool.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "arrc_interfaces/msg/uav_vel_acc.hpp"
// #include "gmm_msgs/msg/gaussian.hpp"
// #include "gmm_msgs/msg/gmm.hpp"
// #include "gmm_msgs/msg/gaussian.hpp"
// #include "gmm_msgs/msg/gmm.hpp"

#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;

//Robots parameters ------------------------------------------------------
const double MAX_ANG_VEL = 0.3;
const double MAX_LIN_VEL = 0.2;         //set to turtlebot max velocities
const double b = 0.025;                 //for differential drive control (only if we are moving a differential drive robot (e.g. turtlebot))
//------------------------------------------------------------------------
const float CONVERGENCE_TOLERANCE = 0.1;
//------------------------------------------------------------------------
const int shutdown_timer = 30;           //count how many seconds to let the robots stopped before shutting down the node


bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

class Controller : public rclcpp::Node
{

public:
    Controller() : Node("distrbuted_gmm_coverage_node")
    {
        //------------------------------------------------- ROS parameters ---------------------------------------------------------
        // ----------- params ----------
        RCLCPP_INFO_STREAM(this->get_logger(), "GMM Coverage constructor called.");
        this->declare_parameter<int>("ROBOTS_NUM", 3);
        this->get_parameter("ROBOTS_NUM", ROBOTS_NUM);
        this->declare_parameter<double>("ROBOT_RANGE", 5.0);
        this->get_parameter("ROBOT_RANGE", ROBOT_RANGE);
        std::cout << "Robots number: " << ROBOTS_NUM << std::endl;
        UAV_NAME = std::getenv("UAV_NAME");
        std::cout << "NAME: " << UAV_NAME << std::endl;
        ID = UAV_NAME[5] - '0';
        std::cout << "I'm UAV " << ID << std::endl;

        // Area parameter
        this->declare_parameter<double>("AREA_SIZE_x", 20);
        this->get_parameter("AREA_SIZE_x", AREA_SIZE_x);
        this->declare_parameter<double>("AREA_SIZE_y", 20);
        this->get_parameter("AREA_SIZE_y", AREA_SIZE_y);
        this->declare_parameter<double>("AREA_LEFT", -10);
        this->get_parameter("AREA_LEFT", AREA_LEFT);
        this->declare_parameter<double>("AREA_BOTTOM", -10);
        this->get_parameter("AREA_BOTTOM", AREA_BOTTOM);

        // ---------- pub/sub ---------
        for (int i = 1; i < ROBOTS_NUM+1; i++)
        {
            odomSubs_.push_back(this->create_subscription<nav_msgs::msg::Odometry>("/Drone" + std::to_string(i) + "/odometry", 1,  [this, i](nav_msgs::msg::Odometry::SharedPtr msg) {this->odomCallback(msg,i);}));
        }
        velPub_ = this->create_publisher<arrc_interfaces::msg::UavVelAcc>("/Drone" + std::to_string(ID) + "/command/setVelocityAcceleration", 1);
        voronoiPub = this->create_publisher<geometry_msgs::msg::PolygonStamped>("/voronoi"+std::to_string(ID)+"_diagram", 1);
        timer_ = this->create_wall_timer(200ms, std::bind(&Controller::loop, this));
        //rclcpp::on_shutdown(std::bind(&Controller::stop,this));

        robots.resize(3, ROBOTS_NUM);

        // GMM params
        means = {{-2.0, -1.0}};
        std::vector<std::vector<float>> single_var = {{0.5, 0.0}, {0.0, 0.5}};

        for (int i = 0; i < means.size(); ++i)
        {
            weights.push_back(1.0/means.size());
            vars.push_back(single_var);
            std::cout << "GMM " << i << ": " << means[i][0] << ", " << means[i][1] << std::endl;
            std::cout << "weight: " << weights[i] << std::endl;
        }
        
    }

    ~Controller()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    //void stop(int signum);
    void stop();
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int j);
    void loop();


    //open write and close LOG file
    void open_log_file();
    void write_log_file(std::string text);
    void close_log_file();


private:
    int ROBOTS_NUM;
    double ROBOT_RANGE;
    int ID;
    std::string UAV_NAME = "Drone1";

    Eigen::MatrixXd robots;
    Eigen::Vector3d p_i;
    
    // ------------------------ ROS params --------------------------------
    rclcpp::Publisher<arrc_interfaces::msg::UavVelAcc>::SharedPtr velPub_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> odomSubs_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr voronoiPub;
    rclcpp::TimerBase::SharedPtr timer_;
    geometry_msgs::msg::Polygon polygon_msg;
    geometry_msgs::msg::PolygonStamped polygonStamped_msg;

    //---------------------------- Environment definition --------------------------------
    double AREA_SIZE_x;
    double AREA_SIZE_y;
    double AREA_LEFT;
    double AREA_BOTTOM;
    
    //------------------------------- GMM params --------------------------------------------
    std::vector<std::vector<float>> means;
    std::vector<std::vector<std::vector<float>>> vars;
    std::vector<float> weights;


    //ofstream on external log file
    std::ofstream log_file;
    long unsigned int log_line_counter=0;
};



void Controller::stop()
{
    RCLCPP_INFO_STREAM(this->get_logger(), "shutting down the controller, stopping the robot.");
    this->timer_->cancel();
    rclcpp::sleep_for(100000000ns);

    RCLCPP_INFO_STREAM(this->get_logger(), "controller has been closed and robot has been stopped");
    rclcpp::sleep_for(100000000ns);
}

void Controller::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int id)
{
    tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    robots.col(id-1) << msg->pose.pose.position.x, msg->pose.pose.position.y, yaw;
}



void Controller::loop()
{
    auto start = this->get_clock()->now().nanoseconds();
    //Parameters
    //double min_dist = 0.4;         //avoid robot collision
    double K_gain = 0.8;                  //Lloyd law gain
    this->log_line_counter = this->log_line_counter + 1;

    //Variables
    double vel_x=0.0, vel_y=0.0, vel_z = 0.0;
    std::vector<Vector2<double>> seeds;
    std::vector<std::vector<float>> centroids;
    std::vector<double> vel; std::vector<float> centroid;

    // ------------------------------------------------------ Environment definition -----------------------------------------------------
    Box<double> AreaBox{AREA_LEFT, AREA_BOTTOM, AREA_SIZE_x + AREA_LEFT, AREA_SIZE_y + AREA_BOTTOM};
    Box<double> RangeBox{-ROBOT_RANGE, -ROBOT_RANGE, ROBOT_RANGE, ROBOT_RANGE};

    p_i = robots.col(ID-1);
    std::cout << "Robot "<< ID << " in " << p_i.transpose() << std::endl;
    std::cout << "All robots : " << robots.transpose() << std::endl;
    

    for (int i = 0; i < ROBOTS_NUM; ++i)
    {
        if (!robots.col(i).isZero(0))
        {
            seeds.push_back({robots(0, i), robots(1, i)});    
        }
        // centroids.push_back({this->pose_x(i), this->pose_y(i)});
    }

    if (!robots.col(ID-1).isZero(0))
    {
        bool robot_stopped = true;

        //-----------------Voronoi--------------------
        //Rielaborazione vettore "points" globale in coordinate locali
        auto local_seeds_i = reworkPointsVector(seeds, seeds[ID-1]);

        // std::cout << "Punto medio gaussiana 1: " << this->gmm_msg.gaussians[0].mean_point.x << ", " << this->gmm_msg.gaussians[0].mean_point.y << std::endl; 
        //Filtraggio siti esterni alla box (simula azione del sensore)
        auto flt_seeds = filterPointsVector(local_seeds_i, RangeBox);
        auto diagram = generateDecentralizedDiagram(flt_seeds, RangeBox, seeds[ID-1], ROBOT_RANGE, AreaBox);
	    // std::cout<<"GOT DIAGRAM\n";
        auto verts = diagram.getVertices();

        this->polygon_msg.points.clear();
        // auto iter = verts.begin();
        // for (int i = 0; i < verts.size(); ++i)
        // {
        //     geometry_msgs::msg::Point32 pt;
        //     std::advance(iter, i); // Move iterator to the i-th position
        //     pt.x = *iter->point.x;
        //     pt.y = *iter->point.y;
        //     this->polygon_msg.points.push_back(pt);
        // }


        // DEBUG
        // std::cout << "Vertici Poligono: \n";
        // for (int i = 0; i < this->polygon_msg.points.size(); ++i)
        // {
        //     std::cout << this->polygon_msg.points[i].x << ", " << this->polygon_msg.points[i].y << std::endl;
        // }

        this->polygonStamped_msg.header.stamp = this->get_clock()->now();
        this->polygonStamped_msg.header.frame_id = "common_origin";
        this->polygonStamped_msg.polygon = this->polygon_msg;
        //compute centroid -- GAUSSIAN DISTRIBUTION
        centroid = computeGMMPolygonCentroid(diagram, this->means, this->vars, this->weights);
        std::cout << "centroid: " << centroid[0] << ", " << centroid[1] << std::endl;
        double norm = sqrt(centroid[0]*centroid[0] + centroid[1]*centroid[1]);
        std::cout << "dist to centroid: " << norm << std::endl;
        if (norm > CONVERGENCE_TOLERANCE)
        {
            std::cout << "ciao\n";
            vel_x = K_gain*(centroid[0]);
            vel_y = K_gain*(centroid[1]);
            vel_z = K_gain*(centroid[2]);
            robot_stopped = false;
        } else {
            vel_x = 0.0;
            vel_y = 0.0;
            vel_z = 0.0;
            std::cout << "ROBOT " << ID << ": STOPPED" << std::endl;
        }

        std::cout<<"sending velocities to " << ID << ":: " << vel_x << ", "<<vel_y<<std::endl;
        arrc_interfaces::msg::UavVelAcc vel_msg;
        vel_msg.header.frame_id = "common_origin";
        vel_msg.velocity.x = vel_x;
        vel_msg.velocity.y = vel_y;
        vel_msg.velocity.z = 0.0;
        vel_msg.acceleration.x = std::nan("1");
        vel_msg.acceleration.y = std::nan("1");
        vel_msg.acceleration.z = std::nan("1");
        vel_msg.yaw = std::nan("1");
        vel_msg.yaw_rate = 0.0;
        velPub_->publish(vel_msg);

        this->voronoiPub->publish(this->polygonStamped_msg);

        auto end = this->get_clock()->now().nanoseconds();
        std::cout<<"Computation time cost: -----------------: "<<end - start<<std::endl;
    } else {
        return;
    }
}



void Controller::open_log_file()
{
    std::time_t t = time(0);
    struct tm * now = localtime(&t);
    char buffer [80];

    char *dir = get_current_dir_name();
    std::string dir_str(dir);

    if (IsPathExist(dir_str + "/GMM_logs"))     //check if the folder exists
    {
        strftime (buffer,80,"/GMM_logs/%Y_%m_%d_%H-%M_logfile.txt",now);
    } else {
        system(("mkdir " + (dir_str + "/GMM_logs")).c_str());
        strftime (buffer,80,"/GMM_logs/%Y_%m_%d_%H-%M_logfile.txt",now);
    }

    std::cout<<"file name :: "<<dir_str + buffer<<std::endl;
    this->log_file.open(dir_str + buffer,std::ofstream::app);
}

void Controller::write_log_file(std::string text)
{
    if (this->log_file.is_open())
    {
        this->log_file << text;
    }
}


void Controller::close_log_file()
{
    std::cout<<"Log file is being closed"<<std::endl;
    this->log_file.close();
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
