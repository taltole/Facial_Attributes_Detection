#!/usr/bin/env python
# coding: utf-8


# ### Imports
from config import *

# # # # # # # # Facial Attributes # # # # # # # # # #

# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# ### Agreement
# The CelebA dataset is available for non-commercial research purposes only.
# All images of the CelebA dataset are obtained from the Internet which are not property of MMLAB, The Chinese
# University of Hong Kong. The MMLAB is not responsible for the content nor the meaning of these images.
# You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes,
# any portion of the images and any portion of derived data.
# You agree not to further copy, publish or distribute any portion of the CelebA dataset. Except, for internal use
# at a single site within the same organization it is allowed to make copies of the dataset.
# The MMLAB reserves the right to terminate your access to the CelebA dataset at any time.
# The face identities are released upon request for research purposes only. Please contact us for details.

# #### Constants


# #### Reading Data
df = pd.read_csv(FILEPATH, index_col='image_id')
df_origin = df.copy()

# #### Counting Nulls
df.isnull().sum().sum()

filenames = df.index.tolist()
print(f'Number of Images:\t{df.shape[0]}')
print(f'Number of Facial att.:\t{df.shape[1]}')

cols = df.columns.tolist()

df.replace(-1, 0, inplace=True)
df.tail()


def zero_labeler(df):
    """
    funtion takes dataframe and add corresponding 0 label for every feature 
    """
    cols = df.columns.tolist()
    for col in cols:
        zero_col = '0_' + col
        df[zero_col] = np.where(df[col] == 0, 1, 0)


zero_labeler(df)


def labels_counter(df, size=20):
    count = df.sum().sort_values(ascending=False)
    print(f'Total Labeled Images:\t{count.sum()}\n')
    pd.DataFrame(count).plot(figsize=(size, size), kind='barh', legend=False)


labels_counter(df);


def plot_labels(df, path, label=None):
    """
    Function takes labeld df and present random image for each label
    if label argument provide than few random images of the same label aer shown
    params: df = dataframe with 1 and 0
            path = for the images folder
            label = name of the label to show (verified by the name of columns)
    returns: images plot
    """

    # verification 
    try:
        assert (df.dtypes != 'O').all()
    except AssertionError:
        print("Please Check Data Frame Labeled correctly with 0 and 1")

    if label is not None and label not in df.columns.tolist():
        raise Exception(f'Error:\tWrong Label Used or not found in dataframe')
        sys.exit()

    # Present Images for One Label
    if label is not None:
        plt.figure(figsize=(10, 10))
        for i in range(1, 21):
            plt.subplot(5, 4, i)
            img = df[label].index[np.random.choice(*np.where(df[label] == 1))]
            image = cv2.imread(find_imagepath(img))
            img = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.suptitle(label)
        plt.show();
        return

    # Present Random Multi_Label Images
    i = 1
    plt.figure(figsize=(25, 25))
    cols = df.columns.tolist()
    ncols = [8 if len(cols) % 8 == 0 else 3]
    shape = np.array(cols).reshape(-1, *ncols)

    for c in cols:
        stop = False
        plt.subplot(shape.shape[0], shape.shape[1], i)
        while not stop:

            try:
                img = df[c].index[np.random.choice(*np.where(df[c] == 1))]
                image = cv2.imread(find_imagepath(img))
                img = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                stop = True
            except Exception as e:
                print(f'Error: {e} with file: {img}')
                break

        plt.title(c)
        plt.xticks([])
        plt.yticks([])
        i += 1

plot_labels(df, IMAGEPATH)

# # Age - Sex - Race
# ##### split datasets to races of different age and sex

# License Claim
# The UTKFace dataset is avaiable for non-commercial research purposes only.
# The aligned and cropped images, as well as landmarks, are obtained by Dlib.
# Please note that all the images are collected from the Internet which are not property
# of AICIP. AICIP is not responsible for the content nor the meaning of these images.
# The copyright belongs to the original owners. If any of the images belongs to you,
# please let us know and we will remove it from our dataset immediately.
# The ground truth of age, gender and race are estimated through the DEX algorithm and
# double checked by a human annotator. If you find anything inaccurate, please let us know.


filenames = os.listdir(IMAGEPATH2)
files_num = len(filenames)


rage_df = pd.DataFrame(index=filenames, columns=['Female', 'Male', 'White', 'Black', 'Asian', 'Indian', 'Latino',
                                                 'Child', 'Teenager', 'Adult', 'Old']).fillna(0)


def df_tagger(df):
    """
    function iterate each row index name and tag 1 for the correct race
    params: dataframe
    returns: races tagged df
    """
    race_dict = {'0': 'White', '1': 'Black', '2': 'Asian', '3': 'Indian', '4': 'Latino'}
    gender_dict = {'0': 'Male', '1': 'Female'}
    ext = ['jpg', 'jpeg', 'png', 'gif', 'tiff']

    for i in df.index:
        if i.split('.')[-1] in ext:

            try:
                # Label Gender
                gender = gender_dict[i.split('_')[SEX]]
                df.loc[i, gender] = 1

                # Label Race
                race = race_dict[i.split('_')[RACE]]
                df.loc[i, race] = 1

                # Label Age
                age = int(i.split('_')[AGE])
                if age <= 10:
                    df.loc[i, 'Child'] = 1
                elif age <= 20:
                    df.loc[i, 'Teenager'] = 1
                elif age <= 60:
                    df.loc[i, 'Adult'] = 1
                elif age <= 120:
                    df.loc[i, 'Old'] = 1
            except:
                pass

    return df.astype(int)


df_tagger(rage_df)

labels_counter(rage_df, 5)

plot_labels(rage_df, IMAGEPATH2)


df3 = pd.read_csv(FILEPATH3)
df_origin = df3.copy()
df3.head()

df3['emotion'].str.lower().value_counts()

# plot_labels(df3, IMAGEPATH3)


# Data Augmentation Class

# ## Reorganize Files to Folders
DFS = pd.concat([df, rage_df])
count = DFS.sum().astype(int)
count.sum()


def summarize_file(df):
    """
    Function get one or many df and create a Summary file with all labels and their related files.
    param: dataframe of a list of dataframes
    returns: df and csv file with all listed file under a label.
    """
    if type(df) == list:
        df = pd.concat(df, join='inner').fillna(0).astype(int)

    files_dict = {}
    cols = df.columns.tolist()
    for col in cols:
        files_dict[col] = df[col][df[col] == 1].index.tolist()
        print(files_dict)
    df_labels = pd.DataFrame.from_dict(files_dict, orient='index').T.fillna('0')
    df_labels.to_csv("files list.csv")
    return df_labels
# summarize_file(DFS)


def rename_images(image_path, keep_name):
    """
    Function takes images path and label's name from source argument to use it to rename files
    according to the labeled folder.
    
    """

    ext = ['jpeg', 'jpg', 'png', 'gif', 'tiff']
    filenames = os.listdir(image_path)
    label = [folder for folder in image_path.split('/') if folder][-1]

    for count, filename in enumerate(filenames):

        if filename.startswith(label):
            pass

        file_ext = filename.split('.')[-1]
        if file_ext in ext:
            if not keep_name:
                dst = f'{label}_{str(count)}.{file_ext}'
            else:
                dst = f'{label}_{filename}'
            src = image_path + filename
            dst = image_path + dst
            os.rename(src, dst)
    print(f"Renaming {count} files Finished!")
# rename_images(IMAGEPATH, True)


def move_copy_images(label, method, path, top, data):
    """
    label - one or list of labels used in data naming
    method - copy, copy2, copytree, move
    path - 
    top - int value to slice files list
    data - filenames origin to extract their name and label
    """
    ext = ['jpeg', 'jpg', 'png', 'gif', 'tiff']

    if data is None:
        dst = input('please provide the name of the folder(or write "label" to use file): ')

    elif data == 'csv':
        with open('files list.csv') as f:
            if isinstance(label, list):
                df = pd.read_csv(f, usecols=label)
            else:
                df = pd.read_csv(f, usecols=[label])

    cols = df.columns.tolist()
    for col in cols:
        dst = col
        #         try:
        #             os.mkdir(col)
        #         except FileExistsError:
        #             pass
        filenames = df.loc[:top, col]

        print(f'Start to {method} files')
        for filename in filenames:
            dst = os.path.join(IMAGEPATH, dst)
            src = os.path.join(IMAGEPATH, str(filename))
            print(f'Source:\t{src}\nTo:\t{dst}')
            file_ext = str(filename).split('.')[-1]
            print(src, dst)
            if file_ext in ext:
                if method == 'copy':
                    # Copy src to dst. (cp src dst) 
                    shutil.copy(src, dst)
                elif method == 'copy2':
                    # Copy files, but preserve metadata (cp -p src dst) 
                    shutil.copy2(src, dst)
                elif method == 'copytree':
                    # Copy directory tree (cp -R src dst) 
                    shutil.copytree(src, dst)
                elif method == 'move':
                    # Move src to dst (mv src dst) 
                    shutil.move(src, dst)  # print(os.path.join(directory, filename))
            else:
                continue
# move_copy_images('Eyeglasses', IMAGEPATH, 5, 'copy', 'csv')