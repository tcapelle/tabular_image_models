from .imports import *
from .models import *
from .data import *

PATH = Path.cwd()
folder = 'zoom15_256'

def in_qrange(ser, q:list=[0, 0.1])->Series:
    "returns serie between q values"
    return ser.between(*ser.quantile(q=q))

def get_top10(df:DataFrame, price_col:str)->DataFrame:
    "Adds a columns of top10/bottom10 do a dataframe"
    price = df[price_col]
    bottom_idx = in_qrange(price, [0,0.1])
    top_idx = in_qrange(price, [0.9,1])    
    df['top10'] = np.nan
    df.loc[top_idx, 'top10']=1
    df.loc[bottom_idx, 'top10']=0
    return df

def get_classifier_df(data, available_labels, price_col='trf_purchprice'):
    "Computes the final df with validation idxs"
    all_ids = pd.merge(data[['cad_id', price_col, 'train']], available_labels[['cad_id', 'fname']], on='cad_id')
    all_ids = get_top10(all_ids, price_col)
    top10_df = all_ids[~all_ids.top10.isna()].reset_index(drop=True)
    return all_ids, top10_df

def get_db(df, test_df, val_idxs, path=PATH, folder=folder, bs=64):
    db = (ImageList.from_df(df, path=path, folder=folder, cols='fname')
          .split_by_idx(list(val_idxs))
          .label_from_df(cols='top10', label_cls=CategoryList)
          .add_test([(PATH/folder)/f for f in  test_df.fname])
          .databunch(bs=bs).normalize(imagenet_stats)
         )
    return db

def create_features(learn, test_df, path='.', head_layers=slice(0,5), feat_suffix='img_feat_'):
    "Cut model head and recover the feature map, returns DataFrame"
    path = Path(path)
    p, t = learn.get_preds()
    print(f'Loading a Classifier with error_rate: {error_rate(p,t)}')
    learn.model = nn.Sequential(learn.model[0], learn.model[1][head_layers])
    p, t = learn.get_preds(DatasetType.Test) #DatasetType.Test
    aux = DataFrame(to_np(p), columns = [(feat_suffix+str(k)) for k in range(p.shape[1])])
    features = pd.concat([test_df['cad_id'], aux], axis=1)
    print(f'features.shape: {features.shape}')
    return features

def train(path='.', imagepath='.', folder='zoom20', arch=models.xresnet34, bs=128, 
          epochs=12, lr=1e-3, mixup=True, low_res=True, save_name='xr34_sample_'):
    path = Path(path)
    task = folder +'_256' if low_res else folder
    intermediate_features = 256
    data = get_data(path)
    IMAGEPATH = imagepath/task
    print(f'Reading images from: {IMAGEPATH}')
    available_labels = get_available_images(IMAGEPATH)
    all_ids, top10_ids = get_classifier_df(data, available_labels)
    val_idxs = top10_ids[~top10_ids['train']].index
    db = get_db(df=top10_ids, test_df=all_ids, val_idxs=val_idxs, path=path, folder=task, bs=bs)
    learn = classification_learner(db, 
                                arch,
                                pretrained=False,
                                head = create_head(512, 2, lin_ftrs=[intermediate_features], concat_pool=False),
                                wd=1)
    learn = learn.mixup().to_fp16() if mixup else learn.to_fp16()
    learn.fit_one_cycle(epochs, lr)
    save_path = learn.save(save_name + task, return_path=True)
    print(f'Model saved on : {save_path}')
    features = create_features(learn, all_ids, path, slice(0,2))
    return features