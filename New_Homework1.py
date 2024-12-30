import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

point_of_lamps = np.array([[4.1, 20.4, 4],[14.1, 21.3, 3.5],[22.6, 17.1, 6],[5.5, 12.3, 4.0],
[12.2, 9.7, 4.0],[15.3, 13.8, 6],[21.3, 10.5, 5.5],[3.9, 3.3, 5.0],[13.1, 4.3, 5.0],[20.3, 4.2, 4.5]])

ldes = np.ones(625)  


grid = 25
m = grid * grid  # number of pixels
n = len(point_of_lamps)  # Number of lamps
A = np.zeros((m, n))

a=point_of_lamps[:,0]
b=point_of_lamps[:,1]
c=point_of_lamps[:,2]


    
for i in range(m):
    for j in range(n):
        distance=np.sqrt((a[j]-(i % grid))**2 + (b[j]-(i // grid))**2 + c[j]**2)
        d=distance**2
        A[i,j]=1/d

#find p 
#@ is used for matrix multiplication

A = (m / np.sum(A))* A
p=np.linalg.inv(A.T @ A) @ A.T @ ldes

ldes_approx= A @ p

# Calculate the RMS error 1 and 2
rms_error_1 = np.sqrt(np.mean((A @ np.ones(n) - ldes) ** 2))
rms_error_2 = np.sqrt(np.mean((ldes - ldes_approx) ** 2))

print("RMS Error 1:", rms_error_1)
print("RMS Error 2:", rms_error_2)

plt.figure(figsize=(10, 5))




plt.subplot(1, 2, 1)
plt.hist(A @ np.ones(len(p)),bins=20,alpha=0.5, color='blue',align='left')
plt.xlim(0.2, 1.8)
plt.ylim(0, 120)
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Histogram for p = 1')


plt.subplot(1, 2, 2)
plt.hist(A@p, bins=20, alpha=0.5, color='green',align='right')
plt.xlim(0.2, 1.8)
plt.ylim(0, 120)
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Histogram for optimization p')

plt.subplots_adjust(wspace=0.2)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
lamp_powers1=A @ np.ones(len(p))
lamp_powers1=lamp_powers1.reshape(grid,grid)
sns.heatmap(lamp_powers1,ax=axes[0])
axes[0].set_ylim(0, grid)



lamp_powers_opt= A@p
lamp_powers_opt=lamp_powers_opt.reshape(grid,grid)
sns.heatmap(lamp_powers_opt,ax=axes[1])
axes[1].set_ylim(0, grid)

for x, y,height in point_of_lamps:
    axes[0].add_patch(plt.Circle((x, y), 0.2, color='black'))
    axes[0].text(x, y, str(height), color='black', ha='right', va='center')
    axes[1].add_patch(plt.Circle((x, y), 0.2, color='black'))
    axes[1].text(x, y, str(height), color='black', ha='right', va='center')
    
axes[0].set_title("Illumination pattern with lamp powers set to 1")
axes[1].set_title("Illumination pattern with optimized lamp powers ")

plt.subplots_adjust(hspace=0.4)