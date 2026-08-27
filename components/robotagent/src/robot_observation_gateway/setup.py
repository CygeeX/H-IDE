from glob import glob

from setuptools import find_packages, setup

package_name = "robot_observation_gateway"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    package_data={"robot_observation_gateway": ["PROTOCOL.lock"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=[
        "google-crc32c>=1.7,<2",
        "grpcio>=1.81.1,<2",
        "protobuf>=6.33.5,<7",
        "pydantic>=2.13,<3",
        "PyYAML>=6.0,<7",
        "zstandard>=0.23,<1",
    ],
    extras_require={"test": ["pytest>=8.4,<9"]},
    zip_safe=False,
    maintainer="dzp",
    maintainer_email="dzp@phibotics.cn",
    description="Bounded RGB-D observation sidecar for the Robot Agent.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "robot-observation-gateway = robot_observation_gateway.cli:main",
            "observation-localization-fixture = "
            "robot_observation_gateway.localization_fixture:main",
        ]
    },
)
