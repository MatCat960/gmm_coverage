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
        this->nh_priv_.getParam("ID", ID);
        this->nh_priv_.getParam("SIM", SIM);
        this->nh_priv_.getParam("GUI", GUI);
        this->nh_priv_.getParam("ROBOT_RANGE", ROBOT_RANGE);
        this->nh_priv_.getParam("GRAPHICS_ON", GRAPHICS_ON);
        this->nh_priv_.getParam("AREA_SIZE_x", AREA_SIZE_x);
        this->nh_priv_.getParam("AREA_SIZE_y", AREA_SIZE_y);
        this->nh_priv_.getParam("AREA_LEFT", AREA_LEFT);
        this->nh_priv_.getParam("AREA_BOTTOM", AREA_BOTTOM);


        //--------------------------------------------------- Subscribers and Publishers ----------------------------------------------------
    if (SIM)
    {
        // std::cout << "SONO IN SIMULAZIONE\n";
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            odomSub_.push_back(nh_.subscribe<nav_msgs::Odometry>("/turtlebot" + std::to_string(i) + "/odom", 100, std::bind(&Controller::odomCallback, this, std::placeholders::_1, i)));
            // velPub_.push_back(nh_.advertise<geometry_msgs::Twist>("/turtlebot" + std::to_string(i) + "/cmd_vel", 1));   
        }
        velPub_.push_back(nh_.advertise<geometry_msgs::TwistStamped>("/turtlebot" + std::to_string(ID) + "/cmd_vel", 1));
    } else
    {
        // std::cout << "NON SONO IN SIMULAZIONE\n";
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            poseSub_.push_back(nh_.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_node/turtle" + std::to_string(i) + "/pose", 100, std::bind(&Controller::poseCallback, this, std::placeholders::_1, i)));                
        }
        velPub_.push_back(nh_.advertise<geometry_msgs::Twist>("/turtle" + std::to_string(ID) + "/cmd_vel", 1));
    }
    
    joySub_ = nh_.subscribe<geometry_msgs::Twist>("/joy_vel", 1, std::bind(&Controller::joy_callback, this, std::placeholders::_1));
    gmmSub_ = nh_.subscribe<gmm_msgs::GMM>("/gaussian_mixture_model", 1, std::bind(&Controller::gmm_callback, this, std::placeholders::_1));
    voronoiPub = nh_.advertise<geometry_msgs::PolygonStamped>("/voronoi"+std::to_string(ID)+"_diagram", 1);
    timer_ = nh_.createTimer(ros::Duration(0.5), std::bind(&Controller::Formation, this));

    // std::cout << "Publishers and Subscribers initialized \n";
    //----------------------------------------------------------- init Variables ---------------------------------------------------------
    pose_x = Eigen::VectorXd::Zero(ROBOTS_NUM);
    pose_y = Eigen::VectorXd::Zero(ROBOTS_NUM);
    pose_theta = Eigen::VectorXd::Zero(ROBOTS_NUM);
    time(&this->timer_init_count);
    time(&this->timer_final_count);
	this->got_gmm = false;//------------------------------------------------------------------------------------------------------------------------------------

    std::cout << "Robot number " << ID << " ready to fly" << std::endl;
    }
    ~Controller()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    //void stop(int signum);
    void stop();
    void test_print();
    void odomCallback(const nav_msgs::Odometry::ConstPtr msg, int j);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr msg, int j);
    void joy_callback(const geometry_msgs::Twist::ConstPtr msg);
    void gmm_callback(const gmm_msgs::GMM::ConstPtr msg);
    void Formation();
    geometry_msgs::Twist Diff_drive_compute_vel(double vel_x, double vel_y, double alfa);


    //open write and close LOG file
    void open_log_file();
    void write_log_file(std::string text);
    void close_log_file();


private:
    int ROBOTS_NUM = 6;
    double ROBOT_RANGE = 10.0;
    int ID = 0;
    bool SIM = true;
    bool GUI;
    bool NotJustStarted;           // Flag per indicare che sono al primo ciclo di esecuzione del nodo
    bool got_gmm;
    double vel_linear_x, vel_angular_z;
    Eigen::VectorXd pose_x;
    Eigen::VectorXd pose_y;
    Eigen::VectorXd pose_theta;
    std::vector<Vector2<double>> seeds_xy;
    int seeds_counter = 0;
    std::vector<std::vector<double>> corners;
    std::vector<double> position;
    std::vector<std::vector<Vector2<double>>> old_positions;                    // vector containing last 10 positions of each robot

    //------------------------- Publishers and subscribers ------------------------------
    std::vector<ros::Publisher> velPub_;
    std::vector<ros::Subscriber> poseSub_;
    std::vector<ros::Subscriber> odomSub_;
    ros::Subscriber joySub_;
    ros::Subscriber gmmSub_;
    ros::Publisher voronoiPub;
    ros::Timer timer_;
    gmm_msgs::GMM gmm_msg;
    geometry_msgs::Polygon polygon_msg;
    geometry_msgs::PolygonStamped polygonStamped_msg;
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

    //graphical view - ON/OFF
    bool GRAPHICS_ON = true;

    //timer - check how long robots are being stopped
    time_t timer_init_count;
    time_t timer_final_count;

    //ofstream on external log file
    std::ofstream log_file;
    long unsigned int log_line_counter=0;
};



void Controller::test_print()
{
    std::cout<<"ENTERED"<<std::endl;
}

void Controller::stop()
{
    //if (signum == SIGINT || signum == SIGKILL || signum ==  SIGQUIT || signum == SIGTERM)
    ROS_INFO("shutting down the controller, stopping the robots, closing the graphics window");
    // if ((GRAPHICS_ON) && (this->app_gui->isOpen())){
    //     this->app_gui->close();
    // }
    // this->timer_->cancel();
    ros::Duration(0.1).sleep();

    geometry_msgs::TwistStamped vel_msg;
    for (int i = 0; i < 100; ++i)
    {
        this->velPub_[0].publish(vel_msg);
    }

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


geometry_msgs::Twist Controller::Diff_drive_compute_vel(double vel_x, double vel_y, double alfa){
    //-------------------------------------------------------------------------------------------------------
    //Compute velocities commands for the robot: differential drive control, for UAVs this is not necessary
    //-------------------------------------------------------------------------------------------------------

    geometry_msgs::Twist vel_msg;
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
    auto timerstart = std::chrono::high_resolution_clock::now();
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

    std::vector<Box<double>> ObstacleBoxes = {};


    for (int i = 0; i < ROBOTS_NUM; ++i)
    {
        if ((this->pose_x(i) != 0.0) && (this->pose_y(i) != 0.0))
        {
            seeds.push_back({this->pose_x(i), this->pose_y(i)});    
        }
        // centroids.push_back({this->pose_x(i), this->pose_y(i)});
    }

    if ((this->pose_x(ID) != 0.0) && (this->pose_y(ID) != 0.0))
    {
        bool robot_stopped = true;

        //-----------------Voronoi--------------------
        //Rielaborazione vettore "points" globale in coordinate locali
        auto local_seeds_i = reworkPointsVector(seeds, seeds[ID]);

        // std::cout << "Punto medio gaussiana 1: " << this->gmm_msg.gaussians[0].mean_point.x << ", " << this->gmm_msg.gaussians[0].mean_point.y << std::endl; 
        //Filtraggio siti esterni alla box (simula azione del sensore)
        auto flt_seeds = filterPointsVector(local_seeds_i, RangeBox);
        auto diagram = generateDecentralizedDiagram(flt_seeds, RangeBox, seeds[ID], ROBOT_RANGE, AreaBox);
	    // std::cout<<"GOT DIAGRAM\n";
        auto verts = diagram.getVertices();

        this->polygon_msg.points.clear();

        if (GUI)
        {
            for (auto v : verts)
            {
                geometry_msgs::Point32 p;
                p.x = this->pose_x(ID) + v.point.x;                 // global position
                p.y = this->pose_y(ID) + v.point.y;                 // global position
                p.z = 0.0;
                this->polygon_msg.points.push_back(p);   
            }
        }
        // DEBUG
        // std::cout << "Vertici Poligono: \n";
        // for (int i = 0; i < this->polygon_msg.points.size(); ++i)
        // {
        //     std::cout << this->polygon_msg.points[i].x << ", " << this->polygon_msg.points[i].y << std::endl;
        // }

        this->polygonStamped_msg.header.stamp = ros::Time::now();
        this->polygonStamped_msg.header.frame_id = "odom";
        this->polygonStamped_msg.polygon = this->polygon_msg;
        //compute centroid -- GAUSSIAN DISTRIBUTION
        centroid = computeGMMPolygonCentroid2(diagram, this->gmm_msg, ObstacleBoxes);

        double norm = sqrt(centroid[0]*centroid[0] + centroid[1]*centroid[1]);
        if (norm > CONVERGENCE_TOLERANCE)
        {
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

        //-------------------------------------------------------------------------------------------------------
        //Compute velocities commands for the robot: differential drive control, for UAVs this is not necessary
        //-------------------------------------------------------------------------------------------------------
        // auto twist_msg = this->Diff_drive_compute_vel(vel_x, vel_y, this->pose_theta(ID));
        geometry_msgs::TwistStamped vel_msg;
        vel_msg.header.stamp = ros::Time::now();
        vel_msg.header.frame_id = "hummingbird" + std::to_string(ID);
        geometry_msgs::Twist twist_msg;
        twist_msg.linear.x = vel_x;
        twist_msg.linear.y = vel_y;

        // calculate orientation to face centroid
        double theta = atan2(centroid[1], centroid[0]);
        double theta_diff = theta - this->pose_theta(ID);
        if (theta_diff > M_PI) {theta_diff = theta_diff - 2*M_PI;}
        twist_msg.angular.z = theta_diff;
        vel_msg.twist = twist_msg;
        //-------------------------------------------------------------------------------------------------------

        std::cout << "sending cmd_vel to " << ID << ":: " << twist_msg.angular.z << ", "<<twist_msg.linear.x;

        this->velPub_[0].publish(vel_msg);
        if (GUI) {this->voronoiPub.publish(this->polygonStamped_msg);}

        if (robot_stopped == true)
        {
            time(&this->timer_final_count);
            if (this->timer_final_count - this->timer_init_count >= shutdown_timer)
            {
                //shutdown node
                std::cout<<"SHUTTING DOWN THE NODE"<<std::endl;
                this->stop();   //stop the controller
                ros::shutdown();
            }
        } else {
            time(&this->timer_init_count);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout<<"Computation time cost: -----------------: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - timerstart).count()<<" ms\n";
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
