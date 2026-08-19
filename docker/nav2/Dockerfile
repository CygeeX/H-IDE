FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-jazzy-nav2-bringup \
      ros-jazzy-nav2-controller \
      ros-jazzy-nav2-planner \
      ros-jazzy-nav2-bt-navigator \
      ros-jazzy-nav2-lifecycle-manager \
      ros-jazzy-nav2-costmap-2d \
      ros-jazzy-nav2-dwb-controller \
      ros-jazzy-dwb-plugins \
      ros-jazzy-nav2-navfn-planner \
      ros-jazzy-nav2-behaviors \
      ros-jazzy-nav2-behavior-tree \
      ros-jazzy-nav2-util \
      ros-jazzy-nav2-map-server \
      ros-jazzy-nav2-velocity-smoother \
      ros-jazzy-nav2-msgs \
      ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY nav2_params.yaml /opt/h-ide-nav2/nav2_params.yaml
COPY office_floor.yaml /opt/h-ide-nav2/maps/office_floor.yaml
COPY office_floor.pgm /opt/h-ide-nav2/maps/office_floor.pgm
COPY nav2_sim_launch.py /opt/h-ide-nav2/nav2_sim_launch.py

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "/opt/h-ide-nav2/nav2_sim_launch.py"]
