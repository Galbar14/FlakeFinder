import ContrastDiff as contrastFinder
from Tkinter import Tk
from tkFileDialog import askdirectory
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os 

def getFolderData(folder, grayscale = True, rgb = True):

    fileList = os.listdir(folder)
    
    grayList = []
    redList = []
    greenList = []
    blueList = []
    
    imageNames = []
    
  
    for file in fileList:
        im = plt.imread(folder + '/' + file)
        if grayscale:
            gray = rgb2gray(im).transpose()
            grayList.append(gray)
        if rgb:
            redList.append(im[:,:,0].transpose())
            greenList.append(im[:,:,1].transpose())
            blueList.append(im[:,:,2].transpose())
            
        imageNames.append(file)
    
    data = {}
    
    if grayscale:
        grayVals, grayLabels = getImageListData(grayList, imageNames)
        data['gray'] = {'Vals': grayVals ,'Labels': grayLabels}
        
    if rgb:
        redVals, redLabels = getImageListData(redList,imageNames)
        blueVals, blueLabels = getImageListData(blueList, imageNames)
        greenVals, greenLabels = getImageListData(greenList, imageNames)
        data['red'] = {'Vals': redVals, 'Labels': redLabels}
        data['blue'] = {'Vals': blueVals, 'Labels': blueLabels}
        data['green'] = {'Vals': greenVals, 'Labels': greenVals}
        
    return data

def getImageListData(imageList, imageNames):
    imageContrast = []
    
    for image in imageList:
        imageContrast.append(contrastFinder.image_mean_contrast(image))
   
    # Sort the data based on contrast
    zippedData = zip(imageContrast, imageNames)
    zippedData.sort()

    return zip(*zippedData)

def makeBarGraph(vals, labels, axes, color, offset, rgb):
    if rgb:
        x = np.arange(0,5*len(labels),5) + offset
    else:
        x = x = np.arange(0,2*len(labels),2)
    rects = axes.bar(x, vals,color=color, width = 1)
    width = rects[0].get_width()
    
    
    for rect in rects:
        height = rect.get_height() 
        axes.text(rect.get_x()+width/2., 1.05*height, 
                '%0.*f' % (2,height),
                ha='center',va='bottom', fontsize=10, color=color)
  
    return rects, x

def plotFolder(folder, grayscale, rgb, title=None):
    


    data = getFolderData(folder, grayscale, rgb)
    if grayscale:
        grayData = data['gray']
    if rgb:
        redData = data['red']
        greenData = data['green']
        blueData = data['blue']
    # Plot the data
    plt.figure()
    
    fig, ax = plt.subplots(figsize=(25,10))
    
    if grayscale:
        grayRects, x  = makeBarGraph(grayData['Vals'], grayData['Labels'], ax, 'gray', 0,rgb)
        rect_width = grayRects[0].get_width()
    if rgb:
        redRects, x = makeBarGraph(redData['Vals'], redData['Labels'], ax, 'red', rect_width,rgb)
        rect_width = redRects[0].get_width()
        greenRects = makeBarGraph(greenData['Vals'], greenData['Labels'], ax, 'green', 2*rect_width,rgb)
        blueRects = makeBarGraph(blueData['Vals'], blueData['Labels'], ax, 'blue', 3*rect_width,rgb)
    
    plt.xticks(x, grayData['Labels'], rotation='vertical', fontsize=14)
    ax.set_xticks(x)
    if title == None:
        plt.title('Contrast Drops for Flakes',fontsize=18)
    else:
        plt.title(title, fontsize=18)
    plt.ylabel('Contrast Difference',fontsize=14)
    
    if rgb:
        ax.legend((grayRects[0], redRects[0], blueRects[0], greenRects[0]),('Grayscale Contrast', 
                   'Red Channel Contrast', 
                   'Blue Channel Contrast',
                  'Green Channel Constrast'), fontsize=14)
        
    else:
        pass
       
        
    plt.show()
    
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
folder = askdirectory() # show an "Open" dialog box and return the path to the selected file
print(folder)

plotFolder(folder = folder, grayscale = True, rgb = True,
           title='Contrast on All Channels')




