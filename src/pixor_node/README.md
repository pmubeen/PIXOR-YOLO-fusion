# PIXOR_NODE - ROS node implementing PIXOR
This ros node subscribes to `kitti_player/hdl64e`, recieving the pointcloud as `Sensor::msg::Pointcloud2`. The recieved point cloud is parsed into a numpy array which is passed to PIXOR. The resulting bounding boxes are superimposed on a birds eye view (BEV) of the pointcloud and published to the topic `pixor_node/bev`.

## Setup
Setup a virtual env with Python 3.8
```sh
python3.8 -m venv env
```
Activate the virtual env
```sh
source env/bin/activate
```
Install the required dependencies
```sh
pip install -r requirements.txt
```
Check if `path-to-env/lib/python3.8/site-packages` is in `$PYTHONPATH`
```sh
echo $PYTHONPATH
```
If not, add it to `$PYTHONPATH` variable
```sh
export PYTHONPATH="path-to-env/lib/python3.8/site-packages:$PYTHONPATH"
```
## Running the node
Running the node requires two flags `--name Name`, the name of the folder with trained model and `--device Device` to specify if cpu or gpu is used.
```sh
cd src
python main.py --name exp1 --device cuda
```
