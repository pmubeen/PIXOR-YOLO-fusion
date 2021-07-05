FUSION_SOURCE420="source devel/setup.bash"
drive=$1
frequency=$2
#Run each ros package
gnome-terminal -e "bash -c \"roscore; exec bash\""
sleep 2
gnome-terminal -e "bash -c \"${FUSION_SOURCE420}; rosrun depthGet depthget; exec bash\""
sleep 2
gnome-terminal -e "bash -c \"${FUSION_SOURCE420}; roslaunch darknet_ros yolo_v3.launch; exec bash\"" 
#sleep 2
gnome-terminal -e "bash -c \"${FUSION_SOURCE420}; rosrun opencv_deal showROI; exec bash\"" 
sleep 2
gnome-terminal -e "bash -c \"${FUSION_SOURCE420}; rosrun pcl_deal pointdeal; exec bash\"" 
sleep 4
${FUSION_SOURCE420}
roslaunch kitti_player kittiplayer_standalone.launch drive:=$drive f:=$frequency