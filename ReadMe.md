
# Marked register digitisation 
 

### Brief
For every UK  election (local or national) polling stations are provided with registers of the eligble voters for the respective polling station. As electors vote, their names are 'marked' to indicate such, ensuring they are not able to cast a vote twice. 

Marked registers are retained by local authorities and can be provided to political parties for a fee. Political parties purchase photocopies of the marked registers and members volunteer their time for data entry, a single ward (a constituency may have 8-12 wards) can take up to 10 hours to process. 

<b>This project seeks to automate the process of digitising marked registers</b>


#### Data protection
All marked registers are protected by GDPR - the purpose of this readme is to demonstrate some of the work without revealing any sensitive information. All additional work is stored in a private repository.


## Summary of digitisation strategy

The plan is to crop each voter from the marked register, keeping record of the order they appear in the register. Given a csv file of the electoral register, use the image order and pytesseracts interpretation of the image text to map images to the correct elector in the csv. Apply machine learning techniques to labelled (voted/ not voted) images to create a model that predicts whether the register has been marked (a vote has been cast).

### Acquire data
1. Convert each page of scanned mark register to png image.
2. Deskew images (using correct_skew_folder.py)
3. Run pytesseract on each page to identify every symbol and the coorindates of their bounding boxes.
4. Apply DBScan to letter coordinates to identify the bottom left coordinate of each line (one voter per line) in the document, crop fixed dimnesions from this point to extract jpeg of each voter.
7. Save cropped images to disk (name image by order).
9. Assign labels by allocating to subfolders (voted, not voted, absentee voter, redundant image etc)
10. Refine crop dimensions (after running preliminary models)

### Map image to elector (notes)
1. Pytesseract is not always accurate, it performs better with a distinct contrast between black and whites. It may be necessary to heighten the images contrast to improve performance (a consistent contrast between images will also help modelling).
2. Regex
3. Translate misclassified symbols based on order (should always be polling number first, B=8 for example).
4. Classify in tranches, map 'easy classifications' with text then use ordering to establish mapping for images more difficult to map.


### Modelling on images (notes)
1. Labels assigned manually, will need to assign a lot of labels, creating interface for one click labelling would be helpful.
2. Crop coordinates and dimensions will need to be optimised. Images may need to be manipulated or recropped to optimise for machine learning.
3. Given the aim is to identify the presence of a horizontal(ish) line, there may be other numeric representations of the image (other than straight array to dataframe conversion) that will result in higher model accuracy.
3. Pytesseract can interpret lines as dashes, this interpretation could be used to establish whether vote has been cast.
4. Probability threshold for classification could be used to classify a majority of voters with edge cases being presented to humans for classification.
5. Reconstruct image using model coefficient of each pixel to determine what pixels models use in assesing image class.
3. Model, model, model...

### Import functions - see script for function details


```python
run register_functions.py
```


```python
#This function applies pytesseract to image, parses output into dataframe
#also applies dbscan to find page split point and cluster characters by line
page=letter_df('example/example_page10.png')
```

## Using DBScan on character coordinates to determine crop point.


```python

import matplotlib.pyplot as plt
import seaborn as sns
```


```python
#contained in letter_df function, repreated here to get info for plotting
page_numbers=page[page.is_digit==True]
pnc=page_numbers[(page_numbers.left>400) & (page_numbers.left<1300)]  #page_numbers_cropped

#run dbscan on digits x axis position
dbscan = DBSCAN(eps=25, min_samples=15)
dbscan.fit(pnc[['left','dummy']])

pnc['labels']=dbscan.labels_

split=(pnc[pnc['labels']==0]['right'].max()+pnc[pnc['labels']==1]['left'].min())/2

page['is_left_side'] = [(i<=split)*1 for i in page['right']]
```


```python
#step by step process illustrated with plots
fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(12, 20), sharey=True, sharex=True)

fig.suptitle('Split page with dbscan', fontsize=40)

sns.scatterplot(x=page.left, y=page.bottom , hue=page.is_digit, ax=ax[0,0])
ax[0,0].set_title('All symbols')

sns.scatterplot(x=page_numbers['left'], y=page_numbers.bottom,ax=ax[0,1])
ax[0,1].set_title('All digits')

ax[1,0].set_title('Clustered digits (cast vote)')
sns.scatterplot(x=pnc['left'], y=pnc.bottom, hue=pnc['labels'], ax=ax[1,0])

ax[1,1].set_title('All symbols + intercept line')
sns.scatterplot(x=page.left, y=page.bottom)
plt.plot([split, split], [page_numbers.bottom.min(), page_numbers.bottom.max()], 'k-', color = 'r')


plt.show()

```


![png](output_6_0.png)



```python
fig =plt.figure(figsize=(10,10))
sns.scatterplot(x=page.left, y=page.bottom, hue=page.line_labels, palette='rainbow')
fig.suptitle('Lines clustered with dbscan', fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
```


![png](output_7_0.png)



```python
example_crop=cropped_images('example/example_page10.png', crop_width=600, ex_left=-60)
image_as_strings= strings_by_line('example/example_page10.png' )
```


```python
#This is me
#The image crop has been altered to remove my polling number and flat number
#The fixed dimensions can be altered to include or exclude polling number, name or house number.

print(image_as_strings[89][4:-5]) #What pytesseract thinks it says

Image.fromarray(example_crop['pic89'][0])
```

    ChadwickRichardDean





![png](output_9_1.png)




```python
#This is my flatmate
#This is what it shows if you vote by post
#There are additional letter symbols that signify 
#proxy voting and other elector categeories

print(image_as_strings[90][4:-5]) #What pytesseract thinks it says

Image.fromarray(example_crop['pic90'][0])
```

    AMehraRulshan





![png](output_10_1.png)




```python
#It also pick up a few things like this which still need to be filtered out before modelling

print(image_as_strings[0]) #What pytesseract thinks it says

Image.fromarray(example_crop['pic0'][0])
```

    PollingStationRegisterfor230519





![png](output_11_1.png)




```python

```


```python

```
