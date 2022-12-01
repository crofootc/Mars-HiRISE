import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os.path
import time
import sys

####################
# GLOBAL VARIABLES #
####################

ROOT_FOLDER = r"C:\Users\cdcro\Documents\GT\Project\Mars\hirise-map-proj-v3_2\hirise-map-proj-v3_2\map-proj-v3_2" # Tested in Initial Data Review
DATA_MAP = r"C:\Users\cdcro\Documents\GT\Project\Mars\hirise-map-proj-v3_2\hirise-map-proj-v3_2\labels-map-proj_v3_2_train_val_test.txt" # Tested in Initial Data Review

#############
# FUNCTIONS #
#############

# -------------------------------------------------------------------------------
# Initial File Prep (read in filenames  and classes) -- Initial Data Review.ipynb
# -------------------------------------------------------------------------------

# get a list of all filenames in the photo folder
def get_filenames(filepath=ROOT_FOLDER):
    return os.listdir(filepath)
    
# per the NASA website they provided a list of all filenames and their respective classes
# after some review it was noted that not all of these photos were not included
# see Initial Data Review.ipynb to see research on the data that was provided
def get_datamap(filepath=DATA_MAP):
    datamap = pd.read_csv(filepath, delimiter=" ", header=None)
    datamap = datamap.rename(columns = {0: "File Name", 1: "Class", 2: "label"})
    return datamap

# add  in details of which files are actually included
def datamap_setup(datamap, filenames):
    datamap['PhotoIncluded'] = 0

    for i, row in datamap.iterrows():
        if row['File Name'] in filenames:
            datamap.loc[i,['PhotoIncluded']]= 1
    
    return datamap

# run all setup functions and provide a list of files we have, and the updated datamap
def initial_setup(filenames_path=ROOT_FOLDER, datamap_path=DATA_MAP):
    filenames = get_filenames(filenames_path)
    datamap = get_datamap(datamap_path)
    datamap = datamap_setup(datamap, filenames)
    return filenames, datamap

# --------------------------------------------------------------------------------------
# Initial image review (get image data from filename, show image) - Initial Image Review
# --------------------------------------------------------------------------------------

# Get image array from file name
def get_image_data(path, debug=False, reduction=None, shrink_dim=None):
    # Reduction variable: (rows to cut off, columns to cutoff)
    # Shrink Variables: (rows, columns) ---> See Data Size Handling
        
    def shrink(data, rows, cols):
        return data.reshape(rows, int(data.shape[0]/rows), cols, int(data.shape[1]/cols)).sum(axis=1).sum(axis=2).astype("int16")
    
    # load the image
    image = Image.open(path)
    if image.mode == "RGBA":
        orig_shape = image.shape
        image = image.convert('RGB')
        if debug:
            print("original shape:", orig_shape)
            print("new shape:", image.shape)
        

    # convert image to numpy array
    # If reducing we will subset
    
    if reduction is None:
        data = np.asarray(image)
    else:
        data = np.asarray(image)[:-reduction[0],:-reduction[1]]

    if shrink_dim is not None:
        # For 225x225 these dims work nicely (smaller is less detail): # 25, 45, 75
        data = shrink(data, shrink_dim[0], shrink_dim[1])
            
    if debug:
        print(type(data))
        # summarize shape
        print(data.shape)
    
    return data
    
def show_image(compressed_image):
    fig, ax0 = plt.subplots()

    ax0.imshow(compressed_image)
    plt.show()
    
def get_image_example(data_map, class_type, loc=0, debug=False, root_path=ROOT_FOLDER, reduction=None):
    # inputting -1 for loc will produce a random image
    if loc == -1:
        loc = np.random.randint(0,datamap[(datamap.PhotoIncluded==1) & (datamap.Class==class_type)]['File Name'].shape[0])
    
    example_image = datamap[(datamap.PhotoIncluded==1) & (datamap.Class==class_type)]['File Name'].iloc[loc]
    
    example_image = os.path.join(root_path,example_image)
    
    show_image(get_image_data(example_image, debug, reduction))
    return example_image

# --------------------------------------------------------------------------------------
# Setting up our data to work with ---- Image Data Setup
# --------------------------------------------------------------------------------------

def data_setup(datamap, reduce=True, img_reduction=None, img_shrink=None):
    df = datamap.copy()
    
    def get_full_path(row):
        return os.path.join(ROOT_FOLDER, row)
    
    def df_get_image_data(row, img_reduction, img_shrink):
        # H W D
        raw_data = get_image_data(row, reduction=img_reduction, shrink_dim=img_shrink)
        return raw_data
    
    def df_image_shape(row):
        return row.shape

    def df_flatten_arr(row):
        r = row.shape[0]
        c = row.shape[1]
                      
        return row.reshape(1,r*c)[0].T
    
    df['Full Path'] = df['File Name'].apply(lambda x: get_full_path(x))
    
    if reduce:
        df = df[datamap["PhotoIncluded"]==1].copy()
        
    df['Raw Data'] = df['Full Path'].apply(lambda x: df_get_image_data(x, img_reduction, img_shrink))
    df['Shape'] = df['Raw Data'].apply(lambda x: df_image_shape(x))
    df['flattened_arr'] = df['Raw Data'].apply(lambda x: df_flatten_arr(x))
    
    return df

# creates numpy array with label in front and then the pic behind
def get_picture_array(pic_data, extra_labels):
    # arr = df['flattened_arr'].to_numpy()
    # cl = df['Class'].to_numpy()
    # arr = np.concatenate(arr, axis=0).reshape(33630,51529)
    # image_arr = get_picture_array(arr, cl)
    
    return np.column_stack((extra_labels, pic_data))

# ------------------------------------------------------------------------------------------------
# Data Size Handling -- produce compressed np arrays of our pixel data -> Data Size Handling.ipynb
# ------------------------------------------------------------------------------------------------

def get_compressed_data(compression=(75,75), debug=False):
    ### GET DATA
    
    start = time.time()

    filenames, datamap = initial_setup()

    
    # A
    mid_a = time.time()
    datamap = data_setup(datamap, reduce=True, img_reduction=(2,2),img_shrink=compression)
 
    # B
    mid_b = time.time()
    # Get picture array
    arr = datamap['flattened_arr'].to_numpy()
    cl = datamap['Class'].to_numpy()

    row = arr.shape[0]
    col = arr[0].shape[0]

    arr = np.concatenate(arr, axis=0).reshape(row,col)
    image_arr = get_picture_array(arr, cl)

    dimensions = datamap.Shape.iloc[0]
    end = time.time()
    
    if debug:
        print("Total Time:", end-start)
        print("Initial SetUp Time:", mid_a - start)
        print("Datamap Time (A):", mid_b - mid_a)
        print("Image Arr Time (B):", end - mid_b)        
    return datamap, image_arr
        