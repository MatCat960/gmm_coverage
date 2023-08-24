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
#include <signal.h>
// SFML
// #include <SFML/Graphics.hpp>
// #include <SFML/OpenGL.hpp>
// My includes
#include "gmm_coverage/FortuneAlgorithm.h"
#include "gmm_coverage/Voronoi.h"
#include "gmm_coverage/Diagram.h"
#include "gmm_coverage/Graphics.h"

// ROS includes
#include "ros/ros.h"
#include <ros/package.h>

#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/PolygonStamped.h>
#include <nav_msgs/Odometry.h>
#include <gmm_msgs/Gaussian.h>
#include <gmm_msgs/GMM.h>

#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;

//Robots parameters ------------------------------------------------------
const double MAX_ANG_VEL = 1.5;
const double MAX_LIN_VEL = 1.5;         //set to turtlebot max velocities
const double b = 0.025;                 //for differential drive control (only if we are moving a differential drive robot (e.g. turtlebot))
//------------------------------------------------------------------------
const float CONVERGENCE_TOLERANCE = 0.1;
//------------------------------------------------------------------------
const int shutdown_timer = 10;           //count how many seconds to let the robots stopped before shutting down the node
sig_atomic_t volatile node_shutdown_request = 0;    //signal manually generated when ctrl+c is pressed


bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

class Controller
{

public:
    Controller() : nh_priv_("~")
    {
        //------------------------------------------------- ROS parameters ---------------------------------------------------------

        this->nh_priv_.getParam("ROBOTS_NUM", ROBOTS_NUM);
        this->nh_priv_.getParam("SIM", SIM);
        this->nh_priv_.getParam("ROBOT_RANGE", ROBOT_RANGE);
        this->nh_priv_.getParam("AREA_SIZE_x", AREA_SIZE_x);
        this->nh_priv_.getParam("AREA_SIZE_y", AREA_SIZE_y);
        this->nh_priv_.getParam("AREA_LEFT", AREA_LEFT);
        this->nh_priv_.getParam("AREA_BOTTOM", AREA_BOTTOM);
        this->nh_priv_.getParam("SAVE_LOGS", SAVE_LOGS);


        //--------------------------------------------------- Subscribers and Publishers ----------------------------------------------------
    if (SIM)
    {
        // std::cout << "SONO IN SIMULAZIONE\n";
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            odomSub_.push_back(nh_.subscribe<nav_msgs::Odometry>("/turtlebot" + std::to_string(i) + "/odom", 100, std::bind(&Controller::odomCallback, this, std::placeholders::_1, i)));
            // velPub_.push_back(nh_.advertise<geometry_msgs::Twist>("/turtlebot" + std::to_string(i) + "/cmd_vel", 1));   
        }
    } else
    {
        // std::cout << "NON SONO IN SIMULAZIONE\n";
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            poseSub_.push_back(nh_.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_node/turtle" + std::to_string(i) + "/pose", 100, std::bind(&Controller::poseCallback, this, std::placeholders::_1, i)));                
        }
    }
    
    joySub_ = nh_.subscribe<geometry_msgs::Twist>("/joy_vel", 1, std::bind(&Controller::joy_callback, this, std::placeholders::_1));
    gmmSub_ = nh_.subscribe<gmm_msgs::GMM>("/gaussian_mixture_model", 1, std::bind(&Controller::gmm_callback, this, std::placeholders::_1));
    timer_ = nh_.createTimer(ros::Duration(0.5), std::bind(&Controller::eval, this));

    if (SAVE_LOGS)
    {
        open_log_file();
    }

    // std::cout << "Publishers and Subscribers initialized \n";
    //----------------------------------------------------------- init Variables ---------------------------------------------------------
    pose_x = Eigen::VectorXd::Zero(ROBOTS_NUM);
    pose_y = Eigen::VectorXd::Zero(ROBOTS_NUM);
    pose_theta = Eigen::VectorXd::Zero(ROBOTS_NUM);
    time(&this->timer_init_count);
    time(&this->timer_final_count);
	this->got_gmm = false;
    area_max = ROBOTS_NUM * M_PI * pow(0.5*ROBOT_RANGE, 2);               // max area covered by the swarm
    //------------------------------------------------------------------------------------------------------------------------------------
    }
    ~Controller()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    //void stop(int signum);
    void stop();
    void odomCallback(const nav_msgs::Odometry::ConstPtr msg, int j);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr msg, int j);
    void joy_callback(const geometry_msgs::Twist::ConstPtr msg);
    void gmm_callback(const gmm_msgs::GMM::ConstPtr msg);
    void eval();


    //open write and close LOG file
    void open_log_file();
    void write_log_file(std::string text);
    void close_log_file();


private:
    int ROBOTS_NUM = 6;
    double ROBOT_RANGE = 10.0;
    bool SIM = true;
    bool GUI;
    bool SAVE_LOGS = false;
    bool NotJustStarted;           // Flag per indicare che sono al primo ciclo di esecuzione del nodo
    bool got_gmm;
    double vel_linear_x, vel_angular_z;
    Eigen::VectorXd pose_x;
    Eigen::VectorXd pose_y;
    Eigen::VectorXd pose_theta;
    std::vector<Vector2<double>> seeds_xy;
    int seeds_counter = 0;

    // ------------------------ Evaluation parameters -----------------------------
    double optim = 0.0;                 // optimality of config (how close to centroidal configuration)
    double effect = 0.0;                // coverage effectiveness (how well drones can measure pdf)
    double eps = 0.0;                   // efficiency (how well drones are covering the environment) --> (track numerator only)
    double area_max;

    //------------------------- Publishers and subscribers ------------------------------
    std::vector<ros::Subscriber> poseSub_;
    std::vector<ros::Subscriber> odomSub_;
    ros::Subscriber joySub_;
    ros::Subscriber gmmSub_;
    ros::Timer timer_;
    gmm_msgs::GMM gmm_msg;
    ros::NodeHandle nh_;
    ros::NodeHandle nh_priv_;
    
    //-----------------------------------------------------------------------------------

    //Rendering with SFML
    //------------------------------ graphics window -------------------------------------
    // std::unique_ptr<Graphics> app_gui;
    //------------------------------------------------------------------------------------

    //---------------------------- Environment definition --------------------------------
    double AREA_SIZE_x = 20.0;
    double AREA_SIZE_y = 20.0;
    double AREA_LEFT = 10.0;
    double AREA_BOTTOM = 10.0;
    //------------------------------------------------------------------------------------

    //---------------------- Gaussian Density Function parameters ------------------------
    bool GAUSSIAN_DISTRIBUTION;
    double PT_X;
    double PT_Y;
    double VAR;

    //------------------------------------------------------------------------------------

    //timer - check how long robots are being stopped
    time_t timer_init_count;
    time_t timer_final_count;

    //ofstream on external log file
    std::ofstream log_file;
    long unsigned int log_line_counter=0;
};



void Controller::stop()
{
    //if (signum == SIGINT || signum == SIGKILL || signum ==  SIGQUIT || signum == SIGTERM)
    ROS_INFO("shutting down the evaluator node");
    if (SAVE_LOGS)
    {
        close_log_file();
    }
    ros::Duration(0.1).sleep();

    ROS_INFO("controller has been closed and robots have been stopped");
    ros::Duration(0.1).sleep();

    ros::shutdown();
}

void Controller::odomCallback(const nav_msgs::Odometry::ConstPtr msg, int j)
{
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

void Controller::poseCallback(const geometry_msgs::PoseStamped::ConstPtr msg, int j)
{
    this->pose_x(j) = msg->pose.position.x;
    this->pose_y(j) = msg->pose.position.z;

    tf2::Quaternion q(
    msg->pose.orientation.x - 0.707,
    msg->pose.orientation.y,
    msg->pose.orientation.z,
    msg->pose.orientation.w + 0.707);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    this->pose_theta(j) = yaw;

    // std::cout << "ROBOT " << std::to_string(ID) << " pose_x: " << std::to_string(this->pose_x(j)) << ", pose_y: " << std::to_string(this->pose_y(j)) << ", theta: " << std::to_string(this->pose_theta(j)) << std::endl;
}

void Controller::joy_callback(const geometry_msgs::Twist::ConstPtr msg)
{
    this->vel_linear_x = msg->linear.x;
    this->vel_angular_z = msg->angular.z;
}

void Controller::gmm_callback(const gmm_msgs::GMM::ConstPtr msg)
{
    this->gmm_msg.gaussians = msg->gaussians;
    this->gmm_msg.weights = msg->weights;
    this->got_gmm = true;
    // std::cout << "Sto ricevendo il GMM\n";
}



void Controller::eval()
{
    if (!this->got_gmm) return;

    // ------------------------------------------------------ Environment definition -----------------------------------------------------
    Box<double> AreaBox{AREA_LEFT, AREA_BOTTOM, AREA_SIZE_x + AREA_LEFT, AREA_SIZE_y + AREA_BOTTOM};
    Box<double> RangeBox{-ROBOT_RANGE, -ROBOT_RANGE, ROBOT_RANGE, ROBOT_RANGE};

    std::vector<Vector2<double>> seeds;
    effect = 0.0;
    double surf = 0.0;
    double optim = 0.0;

    for (int i = 0; i < ROBOTS_NUM; ++i)
    {
        if ((this->pose_x(i) != 0.0) && (this->pose_y(i) != 0.0))
        {
            seeds.push_back({this->pose_x(i), this->pose_y(i)});    
        }
        // centroids.push_back({this->pose_x(i), this->pose_y(i)});
    }

    std::cout << "-------------------------------------\n";
    for (int i = 0; i < ROBOTS_NUM; ++i)
    {
        if (seeds.size() >= 1)
        {
            //-----------------Voronoi--------------------
            //Rielaborazione vettore "points" globale in coordinate locali
            auto local_seeds_i = reworkPointsVector(seeds, seeds[i]);
            
            //Filtraggio siti esterni alla box (simula azione del sensore)
            auto flt_seeds = filterPointsVector(local_seeds_i, RangeBox);
            auto diagram = generateDecentralizedDiagram(flt_seeds, RangeBox, seeds[i], ROBOT_RANGE, AreaBox);

            // Calculate effectiveness of current diagram
            double effect_i = calculateEffectiveness(diagram, this->gmm_msg);
            // std::cout << "Current effectiveness: " << effect_i << std::endl;
            effect = effect + effect_i;

            // Calculate area of current diagram
            double area_i = calculateArea(diagram);
            std::cout << "Current area: " << area_i << std::endl;
            surf = surf + area_i;

            // Centralized voronoi (for config optimality)
            auto diagram_centr = generateCentralizedDiagram(seeds, AreaBox);
            double optim_i = calculateOptim(diagram_centr, this->gmm_msg);
            // std::cout << "Current optimality: " << optim_i << std::endl;
            optim = optim - optim_i;
        }
    }

    eps = surf / area_max; 
    std::cout << "Total effectiveness: " << effect << std::endl;
    std::cout << "Total area: " << surf << std::endl;
    std::cout << "Area coverage efficiency: " << eps << std::endl;
    std::cout << "Optimality of configuration: " << optim << std::endl;
    std::cout << "-------------------------------------\n";

    if (SAVE_LOGS)
    {
        std::string text = std::to_string(effect) + " " + std::to_string(surf) + " " + std::to_string(optim) + "\n";
        write_log_file(text);
    }

}




void Controller::open_log_file()
{
    std::time_t t = time(0);
    struct tm * now = localtime(&t);
    char buffer [80];

    std::string path = ros::package::getPath("gmm_coverage");

    if (IsPathExist(path + "/logs"))     //check if the folder exists
    {
        strftime (buffer,80,"/logs/%Y_%m_%d_%H-%M_logfile.txt",now);
    } else {
        system(("mkdir " + (path + "/logs")).c_str());
        strftime (buffer,80,"/logs/%Y_%m_%d_%H-%M_logfile.txt",now);
    }

    std::cout<<"file name :: "<<path + buffer<<std::endl;
    this->log_file.open(path + buffer,std::ofstream::app);
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
    node_shutdown_request = 1;
}

int main(int argc, char **argv)
{
    signal(SIGINT, nodeobj_wrapper_function);

    ros::init(argc, argv, "centralized_gmm_node", ros::init_options::NoSigintHandler);
    auto node = std::make_shared<Controller>();

    while(!node_shutdown_request)
    {
        ros::spinOnce();
    }
    node->stop();

    if (ros::ok())
    {
        ROS_WARN("ROS HAS NOT BEEN PROPERLY SHUTDOWN, FORCING SHUTDOWN NOW");
        ros::shutdown();
    }

    return 0;
}
