# Facial_Attributes_Detection
CelebFaces Attributes Dataset (CelebA) dataset was the main one used for this project. 
Link to download datasets : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

CelebA is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. 
The images in this dataset cover large pose variations and background clutter. 

CelebA has large diversities, large quantities, and rich annotations, including:
10,177 number of identities,
202,599 number of face images, and
5 landmark locations, 40 binary attributes annotations per image.
<p align="center">
  <img src=https://github.com/taltole/Facial_Attributes_Detection/blob/master/templates/Picture1.png? width="350" alt="accessibility text">
</p>

The dataset and code library can be employ as the training and test sets for the following computer vision tasks (function and files for that can also be found on notebooks):</br>

- face attribute recognition. </br>
- face detection and croping. </br>
- landmark (or facial part) localization. </br>
- face editing & synthesis/augmantation. </br>

</p>

CNN tranfer learning and classic machine learning models (on embedded images) applied for the classifiaction (for GPU/CPU utilization) in order to compare and evalute GDCV used before deployent for each class.

</p>
<p align="center">
  <img src=https://github.com/taltole/Facial_Attributes_Detection/blob/master/templates/Picture2.png? width="650" height="300" alt="accessibility text">
</p>


### Steps to follow: 
- Required Libraries requiement file
- Extract the datasets with .csv file into same folder or change folder path in config.py
- IND_FILE can be created with Notebook/dataset manager to index dataset from celebA or one user provide.
- there are 3 optional main files to run, depends on the task: 
  1. main.py - CNN transfer learning for facial attribute *binary* classifier.
  2. main_multi.py  - CNN transfer learning for facial attribute *multi* classifier.
  3. main_embedding.py - Using embedded images to use with classic machine learning models (file find top performing model using GSCV) 

</p>
<p align="center">
  <img src=https://github.com/taltole/Facial_Attributes_Detection/blob/master/templates/Picture3.png? width="550" height="350" alt="accessibility text">
</p>

- Files run each different training, at final step of training the model, it will be saved into models folder. 
- Give the path of saved Model in config.py  
- For inference run inference.py for local quary or Flask.py to get UI of Model. 
There are several more files and notebooks for you to try them out for different needs (i.e. orginize, ploting, models history also CM and evalutions summary in csv folders)

Thats All. 
Enjoy.
