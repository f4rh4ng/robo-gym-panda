'''
==============
3D Plott result - Panda end effector positioning environment
==============

'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data = np.load('target_coordinate_and_final_status_data.npy')

print(data)


coll=0
reach=0
exceed=0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print('no. of runs:', data.shape[0]-273)

for i in range (data.shape[0]-273):
    xs = data[i,1]
    ys = data[i,2]
    zs = data[i,3]
    if zs >= 0.0:
        if data[i, 0] == 0.:  # collision
            c = 'r'
            m = 'X'
            coll += 1
        elif data[i, 0] == 1.:  # exceeded time
            c = 'b'
            m = 'o'
            exceed+=1
        elif data[i, 0] == 2.:  # success
            c = 'g'
            m = 'p'
            reach+=1
        ax.scatter(xs, ys, zs, c=c, marker=m)


ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()

print('collision:',coll, '  percent:',coll/(data.shape[0]-273),' %')
print('reached:',reach, '  percent:',reach/(data.shape[0]-273),' %')
print('exceed:',exceed,  '  percent:',exceed/(data.shape[0]-273),' %')
