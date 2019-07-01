
# Marked register digitisation 
 

For every UK  election (local or national) polling stations are provided with registers of the eligble voters for the respective polling station. As electors vote, their names are 'marked' to indicate such, ensuring they are not able to cast a vote twice. 

Marked registers are retained by local authorities and can be provided to political parties for a fee. Political parties purchase photocopies of the marked registers and members volunteer their time for data entry, a single ward (a constituency may have 8-12 wards) can take up to 10 hours to process. 

<b>This project seeks to automate the process of digitising marked registers</b>


#### Data protection
All marked registers are protected by GDPR - the purpose of this readme is to demonstrate some of the work without revealing any sensitive information. All additional work is stored in a private repository.

### Summary of digitisation strategy¶
Crop each voter from the marked register, keeping record of the order they appear in the register. Given a csv file of the electoral register, use the image order and pytesseracts interpretation of the image text to map images to the correct elector in the csv. Train machine learning model on labelled (voted/ not voted) images to create a model that predicts whether the register has been marked (a vote has been cast).


[Part 1](https://github.com/richchad/marked-register/blob/master/Cropping%20voters%20from%20marked%20register.ipynb)
Cropping voters

```python

```
