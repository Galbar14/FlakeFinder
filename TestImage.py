import ContrastDiff as contrastFinder
from Tkinter import Tk
from tkFileDialog import askopenfilename
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SAMPLE_ROW = 3

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)

# '/home/gal/code/Spyder/MonolayerID/Img/Ayelet/Image_4200_monolayer_unverified.png'
im = rgb2gray(plt.imread(filename))

# Generate a graph for a sample row
sample_row = im[:,SAMPLE_ROW]
(contrast, filtered) = contrastFinder.signal_contrast(sample_row)

plt.title('Grayscale Image')
sim = im.copy()
sim[:,SAMPLE_ROW] = im.min()
plt.imshow(sim, cmap='Greys_r')

# Pixel Plot
plt.figure()
plt.title('Pixel Intensities Along the X Axis')
plt.xlabel('X Values [Pixels]')
plt.ylabel('Pixel Intensities')

x_cord =  np.arange(sample_row.size)
plt.plot(x_cord,sample_row, 'b-', linewidth=2)
plt.plot(x_cord,filtered, 'g-', linewidth=2)

global_max = filtered.max() 
global_min = filtered.min()

plt.plot(x_cord,np.ones(x_cord.shape[0])*global_max)
plt.plot(x_cord,np.ones(x_cord.shape[0])*global_min)
 
red_patch = mpatches.Patch(color='blue', label='Row Values')
green_patch = mpatches.Patch(color='green', label='Filtered Signal')

plt.legend(handles=[green_patch, red_patch])

# Find the results for all rows and make a graph of that

plt.figure()  
plt.title('Contrast Differences Along Image Rows')
plt.xlabel('Row Number')
plt.ylabel('Contrast Difference')

row_values = contrastFinder.image_contrast_values(im)
plt.plot(np.arange(row_values.size),row_values)

# Find the mean drop
print("TOTAL MEAN DROP: {}".format(row_values.mean()))
