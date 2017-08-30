# -*- coding: utf-8 -*-
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.patches as mpatches
from savitzky_golay import savitzky_golay
from skimage.color import rgb2gray

#PATH = '/home/gal/code/Spyder/MonolayerID/Img/'
#
#FILE_NAME = '#1_num1.xls'
#
#file_path = PATH + FILE_NAME
#
#file = open(file_path, 'rb')
#data = csv.reader(file, delimiter='\t')
#
#table = np.array([row for row in data])
#
#x_cord = table[:,0]
#pix_vals = table[:,1]
#
## Read cross image
##im = rgb2gray(plt.imread('/home/gal/code/Spyder/MonolayerID/Img/cross_#1_1.png'))
#
#from Tkinter import Tk
#from tkFileDialog import askopenfilename
#
#Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
#filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
#print(filename)
#
## '/home/gal/code/Spyder/MonolayerID/Img/Ayelet/Image_4200_monolayer_unverified.png'
#im = rgb2gray(plt.imread(filename))


    
def find_drop(filtered_butter, debug=False):
    ''' Given a signal, find the drop size. '''
    
    x_cord = np.arange(filtered_butter.size)
    # Filter using Butterworth
    

    
    #if debug:
        #plt.title('Filtered Signal')
        #plt.plot(x_cord,filtered_butter)
    # Approximate the filtered signal using a polynomial
    c = np.poly1d(np.polyfit(x_cord, filtered_butter,10))


    crit = c.deriv().r
    r_crit = crit[crit.imag==0].real
    
    # Curviture
    test = c.deriv(2)(r_crit) 

    # compute local minima 
    
    # Add the bounds as potential critical points
    x_min = np.array((c(0)))
    x_min = np.append(x_min,c(filtered_butter.size))

    x_max = np.array((c(0)))
    x_max = np.append(x_max,c(filtered_butter.size))
    
    # excluding range boundaries
    x_min = r_crit[test>0]
    y_min = c(x_min)

    x_max = r_crit[test<0]
    y_max = c(x_max)



    
    # Positive values only
    x_min = x_min[x_min>=0]
    y_min = y_min[y_min>=0]

    x_max = x_max[x_max>=0]
    y_max = y_max[y_max>=0]
    
    # Only before signal ends
    x_min = x_min[x_min<=filtered_butter.size]
    y_min = y_min[y_min<=filtered_butter.size]

    x_max = x_max[x_max<=filtered_butter.size]
    y_max = y_max[y_max<=filtered_butter.size]
    

    # Find the biggest and smallest peak
    
    # Add the bounderies
    y_min = np.append(y_min,c(0))
    y_min = np.append(y_min,c(filtered_butter.size)) 
    
    y_max = np.append(y_max,c(0))
    y_min = np.append(y_max,c(filtered_butter.size)) 
    
 
    global_max = y_max.max()
    global_min = y_min.min()
    
    filtered_signal = c(x_cord)
    global_max = filtered_signal.max()
    global_min = filtered_signal.min()
    
    # Find abs
    drop = global_max-global_min
    # Debug plots
    if debug is True:
        plt.figure()
    
        plt.plot(x_cord,filtered_butter)
        plt.plot(x_cord,filtered_butter, 'r-', linewidth=2)
        plt.plot(x_cord,c(x_cord), 'g-', linewidth=2)
    
        plt.plot(x_cord,np.ones(x_cord.shape[0])*global_max)
        plt.plot(x_cord,np.ones(x_cord.shape[0])*global_min)
        
        plt.title('Pixel intensities along the X axis')
        plt.xlabel('X Values [Pixels]')
        plt.ylabel('Pixel Intensities')
    

    
        #plt.plot( x_min, y_min, 'o' )
        #plt.plot( x_max, y_max, 'o' )
    
        red_patch = mpatches.Patch(color='red', label='Butterworth Filter')
        green_patch = mpatches.Patch(color='green', label='Polynomial Fit')

        plt.legend(handles=[green_patch, red_patch])
    
    return drop
    
def find_line_drops(img, debug):
    ''' Finds the drop size in each row of an image, 
    returns an array of the drop size in row order. '''
    
    drops = np.array([])
    for c in img.transpose():
        drops = np.append(drops,find_drop(c, debug))
    
    return drops

def signal_contrast(pixel_signal):
    N = 2 # Order
    Wn = 0.05 # Cutoff Frequency
    B, A = signal.butter(N, Wn, output='ba')
    filtered_butter = signal.filtfilt(B,A,pixel_signal)\
    
    contrast = find_drop(pixel_signal, debug=False)
    return (contrast, filtered_butter)

def image_contrast_values(image):
    values = np.array([],dtype='float32')
    for row in image:
        row_val = signal_contrast(row)[0]
        values = np.append(values,row_val)
        
        
    return values

def image_mean_contrast(image):
    return image_contrast_values(image).mean()

###### Master Switch ###################
debug = False
 ######################################3

#plt.title('Grayscale Image')
#
#
#drops=find_line_drops(im, debug)
#
#x_cord = x_cord[1:-1].astype(np.float32)
#pix_vals = pix_vals[1:-1].astype(np.float32)
#
#sample = 15
#cross = im[:,sample]
#sim = im.copy()
#sim[:,sample] = im.min()
#plt.imshow(sim, cmap='Greys_r')
#print("Results of one row: {}".format(find_drop(cross,True)))
#
#print("TOTAL MEAN DROP: {}".format(drops.mean()))
#find_drop(pix_vals,debug=True)

#plt.figure()
#plt.title('FFT Power Spectrum of Signal')
#plt.xlabel('Frequency')
#plt.ylabel('Power')
#fft = np.fft.fft(pix_vals)
#fft_shifted = np.fft.fftshift(fft)
#N = fft.size

#val = double(1.0) / N
#w = np.arange(-N/2,N/2, dtype=int)*val*1
#plt.plot(w,np.abs(fft))