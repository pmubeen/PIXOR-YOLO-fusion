#include "tracklets.h"
#include <math.h>

struct object
{
    std::string obj_type;
    float xcoords[8], ycoords[8], zcoords[8];
};

void matmul(float mat1[3][3], float mat2[3][3], float result[3][3]) {
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            result[i][j] = mat1[i][0]*mat2[0][j] + mat1[i][1]*mat2[1][j] + mat1[i][2]*mat2[2][j];
        }
    }
}

float maxfloat(float in[8]) {
    float maxval;
    for (int i = 0; i<8; i++) {
        if (isnan(maxval) || maxval == 0) {
            maxval = in[i];
        }
        else if (maxval < in[i]) {
            maxval = in[i];
        }
    }
    return maxval;
}

float minfloat(float in[8]) {
    float minval;
    for (int i = 0; i<8; i++) {
        if (isnan(minval) || minval == 0) {
            minval = in[i];
        }
        else if (minval > in[i]) {
            minval = in[i];
        }
    }
    return minval;
}

void vecmatmul(float vec[3], float mat[3][3], float result[3]) {
    for (int i=0; i<3; i++) {
        result[i] = vec[0]*mat[0][i] + vec[1]*mat[1][i] + vec[2]*mat[2][i];
    }
}

void rotate_coords(float xcoords[8], float ycoords[8], float zcoords[8], float rotx[3][3], float roty[3][3], float rotz[3][3]) {
    float rotxy[3][3];
    float rotxyz[3][3];
    matmul(rotx, roty, rotxy);
    matmul(rotxy, rotz, rotxyz);
    
    for (int i=0; i<8; i++) {
        float xyz[3] = {xcoords[i], ycoords[i], zcoords[i]};
        float xyzrot[3];
        vecmatmul(xyz,rotxyz,xyzrot);
        xcoords[i] = xyzrot[0]; ycoords[i] = xyzrot[1]; zcoords[i] = xyzrot[2];   
    }
};

std::vector<std::vector<object>> objectsfromFrame(Tracklets *tracklets)
{
    std::vector<std::vector<object>> frames;
    int last_frame;
    int first_frame;
    Tracklets::tTracklet curr_tracklet;
    Tracklets::tPose curr_pose;
    object new_object;
    for (int i = 0; i < tracklets->tracklets.size(); i++)
    {
        curr_tracklet = tracklets->tracklets[i];
        first_frame = curr_tracklet.first_frame;
        last_frame = first_frame + curr_tracklet.poses.size();
        float h, w, l;
        float rx, ry, rz;

        if (frames.size() < last_frame)
        {
            frames.resize(last_frame);
        }

        for (int j = first_frame; j < last_frame; j++)
        {
            curr_pose = curr_tracklet.poses[j - first_frame];
            h = curr_tracklet.h; w = curr_tracklet.w; l = curr_tracklet.l;
            rx = curr_pose.rx; ry = curr_pose.ry; rz = curr_pose.rz;
            new_object.obj_type = curr_tracklet.objectType;

            new_object.xcoords[0] = new_object.xcoords[1] = new_object.xcoords[2] = new_object.xcoords[3] = l / 2;
            new_object.xcoords[4] = new_object.xcoords[5] = new_object.xcoords[6] = new_object.xcoords[7] = -l / 2;

            new_object.ycoords[0] = new_object.ycoords[2] = new_object.ycoords[4] = new_object.ycoords[6] = w / 2;
            new_object.ycoords[1] = new_object.ycoords[3] = new_object.ycoords[5] = new_object.ycoords[7] = -w / 2;

            new_object.zcoords[0] = new_object.zcoords[1] = new_object.zcoords[2] = new_object.zcoords[3] = h / 2;
            new_object.zcoords[4] = new_object.zcoords[5] = new_object.zcoords[6] = new_object.zcoords[7] = -h / 2;

            float rot_x[3][3] = {
                {1, 0, 0},
                {0, cos(rx), -sin(rx)},
                {0, sin(rx), cos(rx)}
            };
            float rot_y[3][3] = {
                {cos(ry), 0, sin(ry)},
                {0, 1, 0},
                {-sin(ry), 0, cos(ry)}
            };
            float rot_z[3][3] = {
                {cos(rz), -sin(rz), 0},
                {sin(rz), cos(rz), 0},
                {0, 0, 1}
            };
            rotate_coords(new_object.xcoords, new_object.ycoords, new_object.zcoords, rot_x, rot_y, rot_z);
            for(int i=0; i<8; i++) {
                new_object.xcoords[i] += curr_pose.tx; new_object.ycoords[i] += curr_pose.ty; new_object.zcoords[i] += curr_pose.tz; 
            }
            frames[j].push_back(new_object);
        }
    }

    return frames;
};