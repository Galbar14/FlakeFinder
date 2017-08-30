# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 08:30:34 2016

@author: gal
"""
import scipy
from scipy import signal

REAL_DROP = 50
RAMP_LENGTH = 500

def nor_cross_corr(a,v):
    ''' Compute the normalized cross correlation between s and t '''
    mean_a = a.mean()
    mean_v = v.mean()
    
    std_v = v.std()
    std_a = a.std()
    
    nor_v = v - mean_v
    nor_a = a - mean_a
    
    nor_factor = 1/(std_a*std_v*a.shape[-1])
    
    # Perform the convolution
    
    rev_nor_v = nor_v[::-1]
    
    convolution = np.convolve(rev_nor_v,nor_a,mode='same')
    
    return convolution*nor_factor
    
def drop_ramp(height, drop, length):
    return np.concatenate((height*np.ones(drop),np.zeros(length-drop)))
    
noise = np.random.normal(0,1,size=RAMP_LENGTH)

ideal_ramp = drop_ramp(REAL_DROP,50,RAMP_LENGTH)

noisy_ramp = np.add(noise, ideal_ramp) 

pix = np.arange(ideal_ramp.shape[0])

plt.plot(pix,noisy_ramp)
plt.title('The Noisy Ramp')

# Go over the expected drop (Maybe 10+-3?) and cross correlate. 
error = 5


correlations = []

plt.figure()
for e in range(-error,error+1):
    guess_ramp = drop_ramp(REAL_DROP+e, 50, RAMP_LENGTH)
    new_cor = nor_cross_corr(guess_ramp,noisy_ramp)
    new_cor_max = new_cor[new_cor==new_cor.max()]
    correlations.append(new_cor_max)
    plt.plot(pix,new_cor)
# Plot guess vs. correlation.
plt.figure()
plt.plot(range(-error,error+1),correlations,'.')
plt.xlabel('Drop error')
plt.ylabel('Max correlation')
