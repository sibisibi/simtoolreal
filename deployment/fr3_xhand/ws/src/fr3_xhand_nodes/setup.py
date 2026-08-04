from setuptools import find_packages, setup

package_name = "fr3_xhand_nodes"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tyler Lum",
    maintainer_email="tylergwlum@gmail.com",
    description="ROS2 nodes for the fr3-xhand deploy stack",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = fr3_xhand_nodes.policy_node:main",
            "goal_node = fr3_xhand_nodes.goal_node:main",
            "hand_node = fr3_xhand_nodes.hand_node:main",
            "fake_robot_node = fr3_xhand_nodes.fake_robot_node:main",
            "fake_perception_node = fr3_xhand_nodes.fake_perception_node:main",
            "home_robot = fr3_xhand_nodes.home_robot:main",
            "arm_bringup = fr3_xhand_nodes.arm_bringup:main",
        ],
    },
)
