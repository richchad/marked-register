#imports
import numpy as np
import cv2
import pytesseract
from PIL import Image
import pandas as pd
import argparse
from sklearn import cluster
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore")

def get_vertical_split(letter_df, eps=25, min_samples=15):
    """Uses DBScan to find the x coordinate to split page in two
    so that all voter names clearly visible on both halfs"""
    #get approx doc width
    doc_width= letter_df['right'].max()
    
    #take centre half of page
    #this crops out the outer columns of numbers
    info= letter_df[(letter_df['left']> .25*doc_width) & (letter_df['left']<.75*doc_width) &
                   (letter_df['is_digit']==1)]
    
    #cluster columns (should result in two clusters)
    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    dbscan.fit(info[['left','dummy']])
    info['labels'] = dbscan.labels_
    
    #check there are only two clusters
    if len(np.unique(dbscan.labels_))>3:
        raise Exception('Error: too many clusters, consider altreing dbscan eps')
    
    #if statements to identify which cluster is on left side
    #use average of letter positions to identify split line
    if info[info['labels']==0]['left'].max()< info[info['labels']==1]['left'].max():
        left = info[info['labels']==0].sort_values(by='right', ascending=False)['right'].iloc[5]
        right= info[info['labels']==1].sort_values(by='left', ascending=True)['left'].iloc[5]
    else:
        left = info[info['labels']==1].sort_values(by='right', ascending=False)['right'].iloc[5]
        right= info[info['labels']==0].sort_values(by='left', ascending=True)['left'].iloc[5]
        
    split= (left+right)/2
    
    return split

def get_line_info(letter_df, eps=7, min_samples=8):
    """Runs DBscan on y coordinates of characters to cluster by line"""
    
    #Set all line labels to not classified
    letter_df['line_labels'] = [-1]*len(letter_df)
    
    #Perform on left side first
    info = letter_df[(letter_df['is_character']==1) & (letter_df['is_left_side']==1)]
    
    #run dbscan on y coordinates only
    dbscan = DBSCAN(eps=eps, min_samples=min_samples) #eps effective between 7 and 11(ish)
    dbscan.fit(info[['top','dummy']])
    
    #order matters - use top coordinate to relabel clusters in order top down
    #assign labels, subset for labelled only
    info['line_labels']=dbscan.labels_
    info=info[info['line_labels']>=0]
    
    #sort values, append cluster label to list in 
    #order of heighest to lowest position on page
    info = info.sort_values(by='top', ascending=False)
    order=[]
    for i in info['line_labels']:
        if i not in order:
            order.append(i)
    
    true_order=range(len(order))
    paired = zip(order, true_order)
    pair_dict={i:j for i,j in paired}
    info['true_line_labels']=[pair_dict[i] for i in info['line_labels']]
    
    #assign lines with ordering to letter_df
    for i in info.index:
        letter_df.loc[i ,'line_labels']= info.loc[i, 'true_line_labels']
    
    #starting line count for right side
    start= max(true_order)+1
    
    #repeat for right side
    info = letter_df[(letter_df['is_character']==1) & (letter_df['is_left_side']==0)]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
    dbscan.fit(info[['top','dummy']])
    info['line_labels']=dbscan.labels_
    info=info[info['line_labels']>=0]
    info = info.sort_values(by='top', ascending=False)
    order=[]
    for i in info['line_labels']:
        if i not in order:
            order.append(i)
    true_order=range(start, start+len(order))
    paired = zip(order, true_order)
    pair_dict={i:j for i,j in paired}
    info['true_line_labels']=[pair_dict[i] for i in info['line_labels']]
    for i in info.index:
        letter_df.loc[i ,'line_labels']= info.loc[i, 'true_line_labels']
    
    return letter_df

def letter_df(image):
    "Input file path of jpeg image"
    #apply pytesseract to png file
    boxes=pytesseract.image_to_boxes(image)
    
    #parse information into data frame
    boxes=boxes.split('\n')
    for i in range(len(boxes)):
        boxes[i]=boxes[i].split(' ')
    data_dict={'letter':[i[0] for i in boxes],
               'left':[int(i[1]) for i in boxes],
               'bottom':[int(i[2]) for i in boxes] ,
               'right': [int(i[3]) for i in boxes],
               'top': [int(i[4]) for i in boxes]}
    boxes = pd.DataFrame(data_dict)
    
    #add dummy column for one variable clustering
    boxes['dummy']=[0]*len(boxes)
    
    #identify all digits in dataframe
    boxes['is_digit']=[(i.isdigit())*1 for i in boxes['letter']]
    
    #identify letter or number
    #bounding boxes of punctuation are not aligned with characters and digits
    #isolating characters and numbers makes identifying lines easier
    boxes['is_character']=[(i.isdigit() | i.isalpha())*1 for i in boxes['letter']]
    
    #x cooridinate to split document on
    split= get_vertical_split(boxes)
    
    #add is_left_side column
    boxes['is_left_side'] = [(i<=split)*1 for i in boxes['right']]
    
    boxes = get_line_info(boxes)
    
    return boxes
              

def cropped_images(image, crop_width=795, crop_height=40, ex_left=15, ex_down=5):
    """Returns dictionary of cropped images and pytesseracts interpretation of cropped image.
    crop width and height are the dimensions of the cropped images
    ex_left: extra left distance from marker to crop (ensure all word is cropped)
    ex_down: extra down distance from marker to crop (ensure all word is cropped)
    """
    
    
    #get letter_df and subset for identified lines
    info=letter_df(image)
    info=info[info['line_labels']>=0]
    
    #get image as array
    image_array = cv2.imread(image)
    
    #setup dictionary for storing images
    images={}
    
    #image height
    height=image_array.shape[0]
    
    #far left crop marker for left and right column
    left_info= round(info[info['is_left_side']==1].groupby(['line_labels']).min()['left'].median())-ex_left
    right_info=round(info[info['is_left_side']==0].groupby(['line_labels']).min()['left'].median())-ex_left
    
    #list of uniqure line labels for left and right column
    left_lines=np.unique(info[info['is_left_side']==1]['line_labels'])
    right_lines=np.unique(info[info['is_left_side']==0]['line_labels'])
    
    #find left and bottom coordinate of each unique line
    for line in left_lines:
        line_info=info[(info.line_labels==line )]
        bottom=height-round(line_info.bottom.median())+ex_down
        images['pic{}'.format(line)]= [image_array[bottom-crop_height:bottom, left_info:left_info+crop_width] ,
                                       line_info]
        
    for line in right_lines:
        line_info=info[(info.line_labels==line )]
        bottom=height-round(line_info.bottom.median())+ex_down
        images['pic{}'.format(line)]= [image_array[bottom-crop_height:bottom, right_info:right_info+crop_width] ,
                                       line_info]
    
    return images

def strings_by_line(image):
    
    df=letter_df(image)
    df=df[df.line_labels>=0].sort_values(by=['line_labels','left'])
    strings=[]

    for line in np.unique(df.line_labels):
        word=''
        for letter in df[(df.line_labels==line)]['letter']:
            word= word +letter
        strings.append(word)
    return strings

