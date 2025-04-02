Camera coordinate system – the coordinate system of most cameras, in which the positive direction of the y-axis points to the ground, the positive direction of the x-axis points to the right, and the positive direction of the z-axis points to the front.

           up  z front
            |    ^
            |   /
            |  /
            | /
            |/
left ------ 0 ------> x right
            |
            |
            |
            |
            v
          y down


LiDAR coordinate system – the coordinate system of many LiDARs, in which the negative direction of the z-axis points to the ground, the positive direction of the x-axis points to the front, and the positive direction of the y-axis points to the left.
虽说我们的雷达是这样的，但是为了契合VAD模型，我们设置lidar为nuscenes的lidar坐标系
             z up  x front
               ^    ^
               |   /
               |  /
               | /
               |/
y left <------ 0 ------ right