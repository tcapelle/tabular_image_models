from .imports import *


def get_features_len(feat_len):
    k=1
    while 2**k <= feat_len:
        feature_len = 2**k
        k += 1
    return feature_len

def get_data(path, train_file='df_train_sample.pkl', valid_file='df_valid_sample.pkl')->DataFrame:
    "concatenates train/valid input files. Create col train."
    df_train = pd.read_pickle(path/train_file).assign(train=True)
    df_valid = pd.read_pickle(path/valid_file).assign(train=False)   
    data = pd.concat([df_train, df_valid])#.reset_index()
    return data

def get_available_images(imagepath)->DataFrame:
    "Computes available images on imagepath folder"
    pattern = r'^cad[0-9]*.jpg\Z'
    available_images = pd.DataFrame([f.name for f in get_image_files(imagepath)], columns=['fname'])
    available_images = available_images[available_images.fname.str.match(pattern)]
    available_labels = available_images.assign(cad_id = lambda x: x.fname.str[3:-4]).astype(dtype={'fname':str, 'cad_id':int})
    return available_labels