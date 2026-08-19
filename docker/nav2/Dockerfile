ARG NAV2_SOURCE_REVISION=6be3614013ec586051b86c97b919b293281490fe

FROM ros:jazzy-ros-base AS nav2-controller-builder

ARG NAV2_SOURCE_REVISION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      python3-colcon-common-extensions \
      ros-jazzy-nav2-controller \
    && rm -rf /var/lib/apt/lists/*

# Jazzy 1.3.12 ignores a failed map->local-frame transform in
# ControllerServer::isGoalReached().  The untouched output pose is all zeroes,
# so a simulator near the map origin can be reported as having arrived exactly
# when its TF publisher dies.  Build the distro-matching package with the
# upstream fail-closed check backported; the commit is verified before patching.
RUN git clone --depth 1 --branch 1.3.12 --single-branch \
      https://github.com/ros-navigation/navigation2.git /opt/navigation2 \
    && test "$(git -C /opt/navigation2 rev-parse HEAD)" = "${NAV2_SOURCE_REVISION}"
COPY controller_server_tf_fail_closed.patch /opt/h-ide-nav2-patches/controller_server_tf_fail_closed.patch
RUN git -C /opt/navigation2 apply --check /opt/h-ide-nav2-patches/controller_server_tf_fail_closed.patch \
    && git -C /opt/navigation2 apply /opt/h-ide-nav2-patches/controller_server_tf_fail_closed.patch \
    && . /opt/ros/jazzy/setup.sh \
    && colcon --log-base /tmp/nav2-log build \
      --base-paths /opt/navigation2/nav2_controller \
      --build-base /tmp/nav2-build \
      --install-base /opt/h-ide-nav2-overlay \
      --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

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
COPY nav2_entrypoint.sh /opt/h-ide-nav2/nav2_entrypoint.sh
COPY --from=nav2-controller-builder /opt/h-ide-nav2-overlay /opt/h-ide-nav2-overlay

ENTRYPOINT ["/opt/h-ide-nav2/nav2_entrypoint.sh"]
CMD ["ros2", "launch", "/opt/h-ide-nav2/nav2_sim_launch.py"]
