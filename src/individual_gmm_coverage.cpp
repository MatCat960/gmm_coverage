// STL
// #include <algorithm>
// #include <cmath>
// #include <cstddef>
// #include <cstdlib>

#include "arrc/coverage/box.h"
#include "arrc/coverage/voronoi_diagram.h"
#include "arrc/coverage/voronoi_fortune_coverage.h"

#include <arrc/common.h>
#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <limits>
#include <map>
#include <math/eigen_converters.h>
#include <math/linalg.h>
#include <memory>
#include <vector>
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

using namespace std::chrono_literals;
namespace gmm_coverage
{
  using namespace arrc::coverage;
  using namespace arrc::math;
  struct Params {
    double robot_range = std::numeric_limits<double>::signaling_NaN();
    double area_width = std::numeric_limits<double>::signaling_NaN();
    double area_height = std::numeric_limits<double>::signaling_NaN();
    double area_left = std::numeric_limits<double>::signaling_NaN();
    double area_bottom = std::numeric_limits<double>::signaling_NaN();
    double convergence_tolerance = std::numeric_limits<double>::signaling_NaN();
    double lloyd_gain = std::numeric_limits<double>::signaling_NaN();
  };
  class GMMController : public rclcpp::Node
  {
  public:
    GMMController() : Node("distrbuted_gmm_coverage_node")
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
      target_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("target", 1);
      voronoiPub = this->create_publisher<geometry_msgs::msg::PolygonStamped>("voronoi_diagram", 1);
      vel_pub_ = this->create_publisher<arrc_interfaces::msg::UavVelAcc>("command/setVelocityAcceleration", 1);
      timer_ = this->create_wall_timer(200ms, std::bind(&GMMController::loop, this));
    }

    ~GMMController() { std::cout << "DESTROYER HAS BEEN CALLED" << std::endl; }

    void declareAndInitParams();
    void initializeGMM();
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void publishVelocity(const Eigen::Vector2d& velocity);
    void neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr msg);
    void loop();

  private:
    // ------------------------ ROS --------------------------------
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr target_pub_;
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
    std::string uav_name_;
    uint32_t uav_id_;
    std::string gps_origin_frame_;
    std::map<int, std::map<int, arrc::Vec2>> team_positions;
    Params params;
    arrc::coverage::Box AreaBox;
    arrc::coverage::Box RangeBox;
    //-------------------------- Coverage -----------------------------------------
    std::vector<arrc::Vec2> seeds;
    std::vector<double> vel;
    std::vector<arrc::Vec2> team_baricenters;
    //------------------------------- GMM params --------------------------------------------
    std::vector<arrc::coverage::Gaussian> gaussians;
  };
  void GMMController::declareAndInitParams()
  {
    uav_name_ = get_namespace();
    uav_name_.erase(0, 1);
    gps_origin_frame_ = uav_name_ + "/gps_origin";
    RCLCPP_INFO(this->get_logger(), "UAV name: %s", uav_name_.c_str());
    // Extract UAV ID from name (format: "Drone{id}")
    try {
      uav_id_ = std::stoul(uav_name_.substr(5)); // Skip "Drone" prefix and convert remaining digits
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to extract UAV ID from name '%s': %s", uav_name_.c_str(), e.what());
      uav_id_ = 0;
    }
    this->declare_parameter<double>("robot_range", 5.0);
    this->declare_parameter<double>("area_width", 20);
    this->declare_parameter<double>("area_height", 20);
    this->declare_parameter<double>("area_left", -10);
    this->declare_parameter<double>("area_bottom", -10);
    this->declare_parameter<double>("lloyd_gain", 0.5);
    this->declare_parameter<double>("convergence_tolerance", 0.5);
    this->declare_parameter<std::vector<double>>("gaussians_x", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_y", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_xx", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_yy", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_xy", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_yx", { 0.0 });
    this->declare_parameter<std::vector<double>>("mix", { 0.0 });
    

    
  }
  void GMMController::initializeGMM()
  {
    auto gaussians_x = this->get_parameter("gaussians_x").as_double_array();
    auto gaussians_y = this->get_parameter("gaussians_y").as_double_array();
    auto gaussians_xx = this->get_parameter("gaussians_xx").as_double_array();
    auto gaussians_yy = this->get_parameter("gaussians_yy").as_double_array();
    auto gaussians_xy = this->get_parameter("gaussians_xy").as_double_array();
    auto gaussians_yx = this->get_parameter("gaussians_yx").as_double_array();
    auto mix = this->get_parameter("mix").as_double_array();
    auto min_common =std::min(mix.size(), std::min(gaussians_x.size(),
                                                   std::min(gaussians_y.size(), std::min(gaussians_xx.size(),
                                                                                         std::min(gaussians_yy.size(), std::min(gaussians_xy.size(), gaussians_yx.size()))))));
    for (size_t i = 0; i < min_common; ++i) {
      Gaussian g((float)mix[i], { (float)gaussians_x[i], (float)gaussians_y[i] },
                 { { (float)gaussians_xx[i], (float)gaussians_xy[i] }, { (float)gaussians_yx[i], (float)gaussians_yy[i] } });
      RCLCPP_INFO(this->get_logger(), "Gaussian %ld: weight = %f, mean = (%f, %f), cov = (%f, %f), (%f, %f)", i, g.weight, g.mean[0], g.mean[1], g.variance[0][0],
                  g.variance[0][1], g.variance[1][0], g.variance[1][1]);
      gaussians.push_back(std::move(g));
    }
  }
  void GMMController::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    odom_ = *msg;
  }
  void GMMController::neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr msg)
  {
    neighbors_.clear();
    std::transform(msg->neighbors.begin(), msg->neighbors.end(), std::back_inserter(neighbors_), [](const nav_msgs::msg::Odometry& neighbor) {
        geometry_msgs::msg::PointStamped point;
        point.header = neighbor.header;
        point.point = neighbor.pose.pose.position;
        return point;
    });
  }
  void GMMController::publishVelocity(const Eigen::Vector2d& velocity)
    {
    arrc_interfaces::msg::UavVelAcc vel_msg;
    vel_msg.header.frame_id = uav_name_ + "/gps_origin";
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
  void GMMController::loop()
  {
    auto start = this->get_clock()->now().nanoseconds();

    this->get_parameter("area_width", params.area_width);
    this->get_parameter("area_height", params.area_height);
    this->get_parameter("area_left", params.area_left);
    this->get_parameter("area_bottom", params.area_bottom);
    AreaBox = arrc::coverage::Box(Vec2f(params.area_left, params.area_bottom),
                                  Vec2f(params.area_left + params.area_width, params.area_bottom + params.area_height));
    this->get_parameter("lloyd_gain", params.lloyd_gain);
    this->get_parameter("convergence_tolerance", params.convergence_tolerance);
    this->get_parameter("robot_range", params.robot_range);
    const auto half_range = params.robot_range / 2.f;
    const Vec2f half_range_vec(half_range);
    RangeBox = arrc::coverage::Box(-half_range_vec, half_range_vec);

    // Variables
    Eigen::Vector2d vel_cmd;
    // ------------------------------------------------------ Environment definition -----------------------------------------------------
    seeds.clear();
    seeds.reserve(neighbors_.size() + 1);
    seeds.push_back({ static_cast<float>(odom_.pose.pose.position.x), static_cast<float>(odom_.pose.pose.position.y) });
    for (auto neighbor : neighbors_) {
        seeds.push_back({ static_cast<float>(neighbor.point.x), static_cast<float>(neighbor.point.y) });
    }

    if (odom_.header.stamp.sec > 0.0) { // be sure we have a valid odometry

      //-----------------Voronoi--------------------
      // Rielaborazione vettore "points" globale in coordinate locali
      auto local_seeds_i = arrc::coverage::reworkPointsVector(seeds, seeds.at(0));

      // std::cout << "Punto medio gaussiana 1: " << this->gmm_msg.gaussians[0].mean_point.x << ", " <<
      // this->gmm_msg.gaussians[0].mean_point.y << std::endl;
      // Filtraggio siti esterni alla box (simula azione del sensore)
      auto flt_seeds = arrc::coverage::filterPointsVector(local_seeds_i, RangeBox);
      auto diagram = arrc::coverage::generateDecentralizedDiagram(flt_seeds, seeds.at(0), params.robot_range, AreaBox);
      auto& verts = diagram.getVertices();

      this->polygon_msg.points.clear();
      for (auto& v : verts) {
        geometry_msgs::msg::Point32 pt;
        pt.x = v->x;
        pt.y = v->y;
        this->polygon_msg.points.push_back(pt);
        RCLCPP_INFO(this->get_logger(), "Vertex: (%f, %f)", pt.x, pt.y);
      }

      this->polygonStamped_msg.header.stamp = this->get_clock()->now();
      this->polygonStamped_msg.header.frame_id = "world";
      this->polygonStamped_msg.polygon = this->polygon_msg;

      // compute centroid -- GAUSSIAN DISTRIBUTION
      auto c = arrc::coverage::computePolygonCentroid(diagram, this->gaussians);
      Eigen::Vector2d centroid = arrc::math::toEigen(c).cast<double>();
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
} // namespace gmm_coverage
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<gmm_coverage::GMMController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
