// STL
#include <algorithm>
#include <arrc_interfaces/msg/detail/neighbors__struct.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <netinet/in.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>
#include <tf2/utils.h>
#include <time.h>
#include <unistd.h>
#include <vector>
// SFML
// #include <SFML/Graphics.hpp>
// #include <SFML/OpenGL.hpp>
// My includes
#include "gmm_coverage/Diagram.h"
#include "gmm_coverage/Voronoi.h"
// #include "Graphics.h"
// ROS includes

#include "arrc_interfaces/msg/neighbors.hpp"
#include "arrc_interfaces/msg/uav_vel_acc.hpp"
#include "rclcpp/rclcpp.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
// #include "gmm_msgs/msg/gaussian.hpp"
// #include "gmm_msgs/msg/gmm.hpp"
// #include "gmm_msgs/msg/gaussian.hpp"
// #include "gmm_msgs/msg/gmm.hpp"

using namespace std::chrono_literals;

bool IsPathExist(const std::string& s)
{
  struct stat buffer;
  return (stat(s.c_str(), &buffer) == 0);
}
struct Params {
  double robot_range = std::numeric_limits<double>::signaling_NaN();
  double area_width = std::numeric_limits<double>::signaling_NaN();
  double area_height = std::numeric_limits<double>::signaling_NaN();
  double area_left = std::numeric_limits<double>::signaling_NaN();
  double area_bottom = std::numeric_limits<double>::signaling_NaN();
  double convergence_tolerance = std::numeric_limits<double>::signaling_NaN();
  double lloyd_gain = std::numeric_limits<double>::signaling_NaN();
};
class Controller : public rclcpp::Node
{
public:
  Controller() : Node("distrbuted_gmm_coverage_node")
  {
    //------------------------------------------------- ROS parameters ---------------------------------------------------------
    // ----------- params ----------
    RCLCPP_INFO(this->get_logger(), "Initializing Distributed GMM Coverage Node.");
    declareAndInitParams();
    initializeGMM();
    // ---------- pub/sub ---------
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odometry", 1, [this](nav_msgs::msg::Odometry::SharedPtr msg) { this->odomCallback(msg); });
    neighbors_sub_ = this->create_subscription<arrc_interfaces::msg::Neighbors>(
        "neighbors_odometry", 1, [this](arrc_interfaces::msg::Neighbors::SharedPtr msg) { this->neighborsCallback(msg); });
    vel_pub_ = this->create_publisher<arrc_interfaces::msg::UavVelAcc>("command/setVelocityAcceleration", 1);
    voronoiPub = this->create_publisher<geometry_msgs::msg::PolygonStamped>("voronoi_diagram", 1);

    timer_ = this->create_wall_timer(200ms, std::bind(&Controller::loop, this));
  }

  ~Controller() { std::cout << "DESTROYER HAS BEEN CALLED" << std::endl; }

  void declareAndInitParams();
  void initializeGMM();
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr msg);
  void publishVelocity(const Eigen::Vector2d& velocity);
  void loop();

private:
  // ------------------------ ROS --------------------------------
  rclcpp::Publisher<arrc_interfaces::msg::UavVelAcc>::SharedPtr vel_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<arrc_interfaces::msg::Neighbors>::SharedPtr neighbors_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr voronoiPub;
  rclcpp::TimerBase::SharedPtr timer_;
  geometry_msgs::msg::Polygon polygon_msg;
  geometry_msgs::msg::PolygonStamped polygonStamped_msg;

  // Odometries
  nav_msgs::msg::Odometry odom_;
  std::vector<geometry_msgs::msg::PointStamped> neighbors_;
  //---------------------------- Environment definition --------------------------------
  std::string uav_name;
  Params params;
  Box<double> AreaBox;
  Box<double> RangeBox;
  //-------------------------- Coverage -----------------------------------------
  std::vector<Vector2<double>> seeds;
  std::vector<double> vel;
  //------------------------------- GMM params --------------------------------------------
  std::vector<std::vector<float>> means;
  std::vector<std::vector<std::vector<float>>> vars;
  std::vector<float> weights;

  // ofstream on external log file
  std::ofstream log_file;
  long unsigned int log_line_counter = 0;
};
void Controller::declareAndInitParams()
{
  this->declare_parameter<double>("robot_range", 5.0);
  this->declare_parameter<double>("area_width", 20);
  this->declare_parameter<double>("area_height", 20);
  this->declare_parameter<double>("area_left", -10);
  this->declare_parameter<double>("area_bottom", -10);
  this->declare_parameter<double>("lloyd_gain", 0.5);
  uav_name = this->get_namespace();
  this->get_parameter("robot_range", params.robot_range);
  this->get_parameter("area_width", params.area_width);
  this->get_parameter("area_height", params.area_height);
  this->get_parameter("area_left", params.area_left);
  this->get_parameter("area_bottom", params.area_bottom);
  this->get_parameter("lloyd_gain", params.lloyd_gain);
}
void Controller::initializeGMM()
{
  // GMM params
  means = { { -2.0, -1.0 } };
  std::vector<std::vector<float>> single_var = { { 0.5, 0.0 }, { 0.0, 0.5 } };

  for (size_t i = 0; i < means.size(); ++i) {
    weights.push_back(1.0 / means.size());
    vars.push_back(single_var);
    std::cout << "GMM " << i << ": " << means[i][0] << ", " << means[i][1] << std::endl;
    std::cout << "weight: " << weights[i] << std::endl;
  }
}
void Controller::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  odom_ = *msg;
}
void Controller::neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr msg)
{
  neighbors_.clear();
  std::transform(msg->neighbors.begin(), msg->neighbors.end(), std::back_inserter(neighbors_), [](const nav_msgs::msg::Odometry& neighbor) {
    geometry_msgs::msg::PointStamped point;
    point.header = neighbor.header;
    point.point = neighbor.pose.pose.position;
    return point;
  });
}
void Controller::loop()
{
  auto start = this->get_clock()->now().nanoseconds();
  log_line_counter++;

  // Variables
  Eigen::Vector2d vel_cmd;
  // ------------------------------------------------------ Environment definition -----------------------------------------------------
  seeds.clear();
  seeds.reserve(neighbors_.size() + 1);
  seeds.push_back({ odom_.pose.pose.position.x, odom_.pose.pose.position.y });
  for (auto neighbor : neighbors_) {
    seeds.push_back({ neighbor.point.x, neighbor.point.y });
  }

  if (odom_.header.stamp.sec > 0.0) { // be sure we have a valid odometry

    //-----------------Voronoi--------------------
    // Rielaborazione vettore "points" globale in coordinate locali
    auto local_seeds_i = reworkPointsVector(seeds, seeds.at(0));

    // std::cout << "Punto medio gaussiana 1: " << this->gmm_msg.gaussians[0].mean_point.x << ", " <<
    // this->gmm_msg.gaussians[0].mean_point.y << std::endl;
    // Filtraggio siti esterni alla box (simula azione del sensore)
    auto flt_seeds = filterPointsVector(local_seeds_i, RangeBox);
    auto diagram = generateDecentralizedDiagram(flt_seeds, RangeBox, seeds.at(0), params.robot_range, AreaBox);
    auto verts = diagram.getVertices();

    this->polygon_msg.points.clear();
    for (auto v : verts) {
      geometry_msgs::msg::Point32 pt;
      pt.x = v.point.x;
      pt.y = v.point.y;
      this->polygon_msg.points.push_back(pt);
    }

    this->polygonStamped_msg.header.stamp = this->get_clock()->now();
    this->polygonStamped_msg.header.frame_id = uav_name + "/gps_origin";
    this->polygonStamped_msg.polygon = this->polygon_msg;
    // compute centroid -- GAUSSIAN DISTRIBUTION
    auto c = computeGMMPolygonCentroid(diagram, this->means, this->vars, this->weights);
    Eigen::Vector2d centroid(c[0], c[1]);
    std::cout << "centroid: " << centroid.transpose() << std::endl;
    double dist = centroid.norm();
    std::cout << "dist to centroid: " << dist << std::endl;
    if (dist > params.convergence_tolerance) {
      vel_cmd = params.lloyd_gain * centroid;
    } else {
      vel_cmd.setZero();
    }
    publishVelocity(vel_cmd);
    this->voronoiPub->publish(this->polygonStamped_msg);
    auto end = this->get_clock()->now().nanoseconds();
    std::cout << "Computation time cost: -----------------: " << end - start << std::endl;
  } else {
    return;
  }
}
void Controller::publishVelocity(const Eigen::Vector2d& velocity)
{
  arrc_interfaces::msg::UavVelAcc vel_msg;
  vel_msg.header.frame_id = uav_name + "/gps_origin";
  vel_msg.velocity.x = velocity.x();
  vel_msg.velocity.y = velocity.y();
  vel_msg.velocity.z = 0.0;
  vel_msg.acceleration.x = std::nan("1");
  vel_msg.acceleration.y = std::nan("1");
  vel_msg.acceleration.z = std::nan("1");
  vel_msg.yaw = std::nan("1");
  vel_msg.yaw_rate = 0.0;
  vel_pub_->publish(vel_msg);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Controller>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
