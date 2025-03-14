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
    std::vector<std::vector<int>> teams;
    std::vector<int> uav_team;
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

      timer_ = this->create_wall_timer(200ms, std::bind(&GMMController::loop, this));
    }

    ~GMMController() { std::cout << "DESTROYER HAS BEEN CALLED" << std::endl; }

    void declareAndInitParams();
    void initializeGMM();
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr msg);
    void loop();

  private:
    // ------------------------ ROS --------------------------------
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr target_pub_;
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
    std::vector<int> uav_team_;
    size_t uav_team_id_;
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
    this->declare_parameter<std::vector<int>>("team_sizes", { 0 });
    this->declare_parameter<std::vector<int>>("team_ids", { 0 });
    this->declare_parameter<std::vector<double>>("gaussians_x", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_y", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_xx", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_yy", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_xy", { 0.0 });
    this->declare_parameter<std::vector<double>>("gaussians_yx", { 0.0 });

    this->get_parameter("robot_range", params.robot_range);
    const auto half_range = params.robot_range / 2.f;

    const Vec2f half_range_vec(half_range);
    RangeBox = arrc::coverage::Box(-half_range_vec, half_range_vec);
    this->get_parameter("area_width", params.area_width);
    this->get_parameter("area_height", params.area_height);
    this->get_parameter("area_left", params.area_left);
    this->get_parameter("area_bottom", params.area_bottom);
    AreaBox = arrc::coverage::Box(Vec2f(params.area_left, params.area_bottom),
                                  Vec2f(params.area_left + params.area_width, params.area_bottom + params.area_height));
    this->get_parameter("lloyd_gain", params.lloyd_gain);
    std::vector<int64_t> team_sizes = get_parameter("team_sizes").as_integer_array();
    std::vector<int64_t> team_ids = get_parameter("team_ids").as_integer_array();

    params.teams.clear();
    params.teams.reserve(team_sizes.size());
    size_t id_idx = 0;
    for (size_t i = 0; i < team_sizes.size(); i++) {
      std::vector<int> team;
      for (size_t j = 0; j < static_cast<size_t>(team_sizes[i]); j++) {
        if (id_idx < team_ids.size()) {
          team.push_back(team_ids[id_idx++]);
        }
      }
      params.teams.push_back(team);
    }

    // Find my team
    uav_team_.clear();
    size_t counter = 0;
    for (const auto& team : params.teams) {
      if (std::find(team.begin(), team.end(), uav_id_) != team.end()) {
        uav_team_ = team;
        uav_team_id_ = counter;
        break;
      }
      counter++;
    }

    if (uav_team_.empty()) {
      RCLCPP_WARN(this->get_logger(), "UAV ID %d not found in any team!", uav_id_);
    } else {
      RCLCPP_INFO_STREAM(this->get_logger(), fmt::format("UAV team: {}", fmt::join(uav_team_, ",")));
    }
  }
  void GMMController::initializeGMM()
  {
    auto gaussians_x = this->get_parameter("gaussians_x").as_double_array();
    auto gaussians_y = this->get_parameter("gaussians_y").as_double_array();
    auto gaussians_xx = this->get_parameter("gaussians_xx").as_double_array();
    auto gaussians_yy = this->get_parameter("gaussians_yy").as_double_array();
    auto gaussians_xy = this->get_parameter("gaussians_xy").as_double_array();
    auto gaussians_yx = this->get_parameter("gaussians_yx").as_double_array();

    auto min_common =
        std::min(gaussians_x.size(),
                 std::min(gaussians_y.size(), std::min(gaussians_xx.size(),
                                                       std::min(gaussians_yy.size(), std::min(gaussians_xy.size(), gaussians_yx.size())))));
    for (size_t i = 0; i < min_common; ++i) {
      Gaussian g((float)1.f / min_common, { (float)gaussians_x[i], (float)gaussians_y[i] },
                 { { (float)gaussians_xx[i], (float)gaussians_xy[i] }, { (float)gaussians_yx[i], (float)gaussians_yy[i] } });
      RCLCPP_INFO(this->get_logger(), "Gaussian %ld: mean = (%f, %f), cov = (%f, %f), (%f, %f)", i, g.mean[0], g.mean[1], g.variance[0][0],
                  g.variance[0][1], g.variance[1][0], g.variance[1][1]);
      gaussians.push_back(std::move(g));
    }
  }
  void GMMController::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    odom_ = *msg;
    team_positions[uav_team_id_][uav_id_] = { (float)odom_.pose.pose.position.x, (float)odom_.pose.pose.position.y };
  }
  void GMMController::neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr msg)
  {
    neighbors_.clear();
    std::transform(msg->neighbors.begin(), msg->neighbors.end(), std::back_inserter(neighbors_),
                   [this](const nav_msgs::msg::Odometry& neighbor) {
                     geometry_msgs::msg::PointStamped point;
                     point.header = neighbor.header;
                     point.point = neighbor.pose.pose.position;
                     // Extract neighbor's ID from frame_id
                     std::string frame_id = neighbor.header.frame_id;
                     size_t start = frame_id.find("Drone") + 5; // Skip "Drone"
                     size_t end = frame_id.find("/", start);
                     int neighbor_id = std::stoi(frame_id.substr(start, end - start));
                     int neighbor_team_id = 0;
                     for (const auto& team : params.teams) {
                       if (std::find(team.begin(), team.end(), neighbor_id) != team.end()) {
                         break;
                       }
                       neighbor_team_id++;
                     }
                     arrc::Vec2 neighbor_pos{ (float)point.point.x, (float)point.point.y };
                     // Add neighbor position to its team's positions
                     team_positions[neighbor_team_id][neighbor_id] = neighbor_pos;
                     return point;
                   });
  }
  void GMMController::loop()
  {
    auto start = this->get_clock()->now().nanoseconds();

    std::map<int, arrc::Vec2> team_baricenters;
    for (const auto& [team_id, positions] : team_positions) {
      size_t team_size = 0;
      arrc::Vec2 team_baricenter{ 0.0f, 0.0f };
      for (const auto& [id, position] : positions) {
        team_baricenter.x += position.x;
        team_baricenter.y += position.y;
        team_size++;
      }
      // Compute average if team has members
      if (team_size > 0) {
        team_baricenter.x /= team_size;
        team_baricenter.y /= team_size;
        team_baricenters[team_id] = team_baricenter;
      }
    }

    // Variables
    Eigen::Vector2d vel_cmd;
    // ------------------------------------------------------ Environment definition -----------------------------------------------------
    seeds.clear();
    seeds.push_back(team_baricenters[uav_team_id_]);
    for (const auto& [id, barycenter] : team_baricenters) {
      if (static_cast<size_t>(id) != uav_team_id_) {
        seeds.push_back(team_baricenters[id]);
      }
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
      Eigen::Vector2d centroid = arrc::math::toEigen(c + team_baricenters[uav_team_id_]).cast<double>();
      std::cout << "centroid: " << centroid.transpose() << std::endl;
      double dist = centroid.norm();
      std::cout << "dist to centroid: " << dist << std::endl;
      nav_msgs::msg::Odometry target_msg;
      target_msg.header.frame_id = gps_origin_frame_;
      target_msg.header.stamp = this->get_clock()->now();
      target_msg.pose.pose.position.x = centroid.x();
      target_msg.pose.pose.position.y = centroid.y();
      target_msg.pose.pose.position.z = 0.0;
      target_msg.pose.pose.orientation.w = 1.0;
      target_msg.pose.pose.orientation.x = 0.0;
      target_msg.pose.pose.orientation.y = 0.0;
      target_msg.pose.pose.orientation.z = 0.0;
      target_pub_->publish(target_msg);
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
