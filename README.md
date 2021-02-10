# Facial_Attributes_Detection
CelebA dataset was the main one used for this project. 
Link to download datasets : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. 
The images in this dataset cover large pose variations and background clutter. 

CelebA has large diversities, large quantities, and rich annotations, including:
10,177 number of identities,
202,599 number of face images, and
5 landmark locations, 40 binary attributes annotations per image.

The dataset can be employed as the training and test sets for the following computer vision tasks: 
- face attribute recognition, 
- face detection, 
- landmark (or facial part) localization
- face editing & synthesis.

### Steps to follow: 
- Required Libraries requiement file
- Extract the datasets with .csv file into same folder or change folder path in config.py
- IND_FILE can be created with Notebook/dataset manager to index dataset from celebA or one user provide.
- there are 3 main files to run: 
  1. main.py - CNN transfer learning for facial attribute *binary* classifier.
  2. main_multi.py  - CNN transfer learning for facial attribute *multi* classifier.
  3. main_embedding.py - Using embedded images to use with machine learn classic models (file find top performing model using GSCV) 

- Run each of files, at final step of Training the model, save that model into models folder. 
- Give the path of saved Model in config.py  
- For inference run inference.py for local quary or Flask.py to get UI of Model. 
There are several more files and notebooks for you to try them out for different needs (i.e. orginize, ploting, models history also CM and evalutions summary in csv folders)

Thats All. 
Enjoy.
