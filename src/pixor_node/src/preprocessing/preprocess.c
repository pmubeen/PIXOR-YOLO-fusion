#include <stdio.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <iomanip>  
#include <vector>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <fstream>    
#include <string>
#include <vector>

using namespace std;
extern "C" {
	const float X_MIN = 0.0;
	const float X_MAX = 70.0;
	const float Y_MIN =-40.0;
	const float Y_MAX = 40.0;
	const float Z_MIN = -2.5; 
	const float Z_MAX = 1;
	const float X_DIVISION = 0.1;
	const float Y_DIVISION = 0.1;
	const float Z_DIVISION = 0.1;
	
	inline int getX(float x){
		return (int)((x-X_MIN)/X_DIVISION);
	}

	inline int getY(float y){
		return (int)((y-Y_MIN)/Y_DIVISION);
	}

	inline int getZ(float z){
		return (int)((z-Z_MIN)/Z_DIVISION);
	}
	
	inline int at3(int y, int x, int z){
		return y * (700 * (35+1)) + x * (35+1) + z;
	} 
	
	inline int at2(int y, int x){
		return y*700 + x;
	}
	
	void decode(unsigned char *data, int ix){
		uint32_t u;
		float f;
		
		u = (data[0] << 0) | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
		std::memcpy(&f, &u, sizeof(float));
		
		printf("data: %u \n", u);
		printf("formatted: %f \n", f);
	}
	
	void processBEV(const void *nparray, unsigned char *data, uint32_t height, uint32_t width, uint32_t point_step, uint32_t row_step){
		float *data_cube = (float *) nparray;
		
		int offset;
		
		//x y z i
		uint32_t u1;
		uint32_t u2;
		uint32_t u3;
		uint32_t u4;
		
		float f1;
		float f2;
		float f3;
		float f4;
	
		for(size_t i = 0; i < height; i++){
			offset = row_step * i;
			
			for(size_t j = 0; j < width; j++){
				//printf("offset: %i \n", offset);
			
				//parse the unsigned integer from the data
				//format is <fffxxxxf (little endian, 3 floats, 4 padding, 1 float total size = 32 bytes)
				u1 = (data[offset + 0] << 0) | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24);
				u2 = (data[offset + 4] << 0) | (data[offset + 5] << 8) | (data[offset + 6] << 16) | (data[offset + 7] << 24);
				u3 = (data[offset + 8] << 0) | (data[offset + 9] << 8) | (data[offset + 10] << 16) | (data[offset + 11] << 24);
				u4 = (data[offset + 28] << 0) | (data[offset + 29] << 8) | (data[offset + 30] << 16) | (data[offset + 31] << 24);
				
				//write the unsigned int to a float
				std::memcpy(&f1, &u1, sizeof(float));
				std::memcpy(&f2, &u2, sizeof(float));
				std::memcpy(&f3, &u3, sizeof(float));
				std::memcpy(&f4, &u4, sizeof(float));
				
				//printf("x: %f \n", f1);
				//printf("y: %f \n", f2);
				//printf("z: %f \n", f3);
				//printf("i: %f \n", f4);
				
				if(f1 > X_MIN && f2 > Y_MIN && f3 > Z_MIN && f1 < X_MAX && f2 < Y_MAX && f3 < Z_MAX){
					int x = getX(f1);
					int y = getY(f2);
					int z = getZ(f3);
					
					*(data_cube + at3(y, x, z)) = 1; 
					*(data_cube + at3(y, x, 35)) += f4; //leave here for now instead of using denisty map for normalization
				}
				
				offset += point_step;
			}	
		}
	}
}