from glob import glob

from setuptools import find_packages, setup


package_name = "robot_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/schemas", glob("schemas/*.json")),
    ],
    install_requires=[
        "jsonschema>=4.25,<5",
        "pydantic>=2.13,<3",
        "PyYAML>=6.0,<7",
        "websockets>=16.1,<17",
    ],
    zip_safe=False,
    maintainer="dzp",
    maintainer_email="dzp@phibotics.cn",
    description="Restricted WebSocket-to-ROS 2 robot-side protocol agent.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["robot-agent = robot_agent.cli:main"]},
)
