from .imports  import *

def rmspe(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "RMSPE between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    pct_var = (targ - pred)/targ
    return torch.sqrt((pct_var**2).mean())

def exp_rmse(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Exp RMSE between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    pred, targ = torch.exp(pred), torch.exp(targ)
    pct_var = (targ - pred)
    return torch.sqrt((pct_var**2).mean())

def L1Flat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    return FlattenedLoss(nn.L1Loss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

class L1WeightedLoss(Module):
    def forward(self, input, target): 
        return ((input-target).abs()/target.abs()).mean()
    
def L1FlatW(*args, axis:int=-1, floatify:bool=True, **kwargs):
    return FlattenedLoss(L1WeightedLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

def df2array(df, feat=None):
    if feat is None:
        return df.values[:, None]
    for i, ch in enumerate(df[feat].unique()):
        data_i = df[df[feat] == ch].values[:, None]
        if i == 0: data = data_i
        else: data = np.concatenate((data, data_i), axis=1)
    return data

def totensor(arr, **kwargs):
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    elif not isinstance(arr, torch.Tensor):
        print(f"Can't convert {type(arr)} to torch.Tensor")
    return arr.float()

def toarray(arr):
    if isinstance(arr, torch.Tensor):
        arr = np.array(arr)
    elif not isinstance(arr, np.ndarray):
        print(f"Can't convert {type(arr)} to np.array")
    if arr.dtype == 'O': arr = np.array(arr, dtype=np.float32)
    return arr

def to3dtensor(arr):
    if arr.dtype == 'O': arr = np.array(arr, dtype=np.float32)
    arr = totensor(arr)
    if arr.ndim == 1: arr = arr[None, None]
    elif arr.ndim == 2: arr = arr[:, None]
    elif arr.ndim == 4: arr = arr[0]
    assert arr.ndim == 3, 'Please, review input dimensions'
    return arr

def to2dtensor(arr):
    if arr.dtype == 'O': arr = np.array(arr, dtype=np.float32)
    arr = totensor(arr)
    if arr.ndim == 1: arr = arr[None]
    elif arr.ndim == 3: arr = torch.squeeze(arr, 0)
    assert arr.ndim == 2, 'Please, review input dimensions'
    return arr

def to3darray(arr):
    arr = toarray(arr)
    if arr.ndim == 1: arr = arr[None, None]
    elif arr.ndim == 2: arr = arr[:, None]
    elif arr.ndim == 4: arr = arr[0]
    assert arr.ndim == 3, 'Please, review input dimensions'
    return np.array(arr)

def verify_against_files(image_list, image_path):
    image_set =set([f.name for f in image_path.ls()])
    inter = set(image_list).intersection(image_set)
    print(f'Image List {len(image_list)}, available images: {len(inter)}')
    return list(inter)



def in_qrange(ser, q:list=[0, 0.1])->Series:
    "returns serie between q values"
    return ser.between(*ser.quantile(q=q))

def results2df(p,t, index):
    return DataFrame(np.exp(np.stack([to_np(p.squeeze()), to_np(t.squeeze())], axis=1)), 
          columns=['p', 't'],
          index=index)

def rmspe_df(df):
    p = torch.from_numpy(df.p.values)
    t = torch.from_numpy(df.t.values)
    return to_np(rmspe(p,t))

def compute_rmspe(p, t, index, qs=[0, 0.1, 0.2, 0.3, 0.4, 1.0]):
    df = results2df(p, t, index)
    q_ranges = zip(qs[0:-1], qs[1:])
    res = []
    for qi, qf in  q_ranges:
        idxs = in_qrange(df.t, [qi, qf])
        res += [rmspe_df(df[idxs]).item()]
    return res

def print_stats(p,t): 
    AE_labels=['A','B','C','D','E']

    p_np = np.array(p.exp().squeeze())
    t_np = np.array(t.exp().squeeze())


    print("")
    print(f'RMSE (log y): {rmse(p,t)}')
    print(f'RMSE : {rmse(p.exp(), t.exp())}')
    print(f'RMSPE (log y): {rmspe(p,t)}')
    print(f'RMSPE: {exp_rmspe(p,t)}')

    NN_AE = pd.cut(np.abs( (p_np/t_np)-1), [0,0.1, 0.2, 0.3, 0.4,9999999], right=True,labels=AE_labels)
    print("")
    print(f'Data size : {NN_AE.shape}')
    print("Performance Accuracy (A: Within 10%, B: Within 20%, etc)")
    print(pd.value_counts(NN_AE,normalize=True).sort_index())
    
    AB_labels=['1: % Error < 20%','2: % Error > 20%']
    NN_AB = pd.cut(np.abs( (p_np/t_np)-1), [0,0.2, 9999999], right=True,labels=AB_labels)
    print("")
    print(f'Data Set size : {NN_AB.shape}')
    print("Performance Accuracy")
    print(pd.value_counts(NN_AB,normalize=True).sort_index())
    
    overunder=np.array(p_np<t_np)
    
    print("")
    print('Summary:')
    print(f'RMSE : {rmse(p.exp(), t.exp())}')
    print(f'RMSPE: {exp_rmspe(p,t)}')
    print(f'%A: {pd.value_counts(NN_AE,normalize=True).sort_index()[0]}')
    print(f'%AB: {pd.value_counts(NN_AB,normalize=True).sort_index()[0]}')
    print('Proportion Under Valuations: '+str(overunder.sum()/overunder.size))
