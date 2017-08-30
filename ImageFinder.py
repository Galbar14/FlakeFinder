# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:43:52 2016

@author: gal
"""

import numpy as np
from scipy import signal

CONTRAST_THRESH = 5
PATH = '/home/gal/code/Spyder/MonolayerID/Img/'

def Find_Canidate_Cordinates(image):
    contrast = Generate_Contrast_Image(image)
    shapes = Find_Canidate_Shapes(contrast)
    diff = Find_Contrast_Difference(shapes)
    
    # Find all contrast differences lower than the 
    # threshold
    
    # Get the right shapes
    
    # Find the shapes' relative cordinates 
    # (Cordinates in the image)
    
def Generate_Contrast_Image(image):
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    
    colvolved = signal.convolve2d(image,kernel,mode='same')
    return cv2.cvtColor(convolved,cv2.COLOR_BGR2GRAY)

def Find_Canidate_Shapes(contrast_img):
    ''' Returns a dictionary of number -> shape.
    For now numbers are abritery. '''
    pass

def Find_Contrast_Difference(number2shape):
    ''' Returns a dictionary of number -> CD. '''
    pass

