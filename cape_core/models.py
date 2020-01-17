from .imports import *

def get_features_map(m, pool=nn.AdaptiveAvgPool2d, flat=False):
    "cut model at pooling layer"
    def has_pool_type(m):
        if is_pool_type(m): return True
        for l in m.children():
            if has_pool_type(l): return True
        return False
    ll = list(enumerate(m.children()))
    cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    layers = list(m.children())[:cut]
    if flat: layers += [pool(1), Flatten()]
    features_map = nn.Sequential(*layers); features_map
    print(f'cut={cut}, pool={pool}')
    return features_map

def classification_learner(db, arch, pretrained=False, head=None, metrics=[error_rate], **kwargs):
    "Gets a classification learner for feature extraction"
    if not pretrained:
        body = get_features_map(arch(pretrained=pretrained))
        num_features = num_features_model(body)
        head = ifnone(head, create_head(2*num_features, db.c))
        model = nn.Sequential(body, head)
        return Learner(db, model, metrics=metrics, **kwargs)
    else: 
        return cnn_learner(db, arch, metrics=metrics, **kwargs)

class TabularFeatureModel(TabularModel):
    def __init__(self, n_features, *args, **kwargs):
        self.n_features = n_features
        super().__init__(*args, **kwargs)
        self.bn_features = nn.BatchNorm1d(n_features)
        
    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont+self.n_features] + layers + [out_sz]
    
    def forward(self, x_tabular:Tensor, x_feats:Tensor) -> Tensor:
        x_cat, x_cont = x_tabular
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        if self.n_features != 0:
            x_feats = self.bn_features(x_feats.view(x_feats.size(0), -1)) #a Flatten
            x = torch.cat([x, x_feats], 1)
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x

def tabular_feature_learner(data:DataBunch, 
                            layers:Collection[int], 
                            emb_szs:Dict[str,int]=None, 
                            ps:Collection[float]=None, 
                            emb_drop:float=0., 
                            y_range:OptRange=None, 
                            use_bn:bool=True, 
                            **learn_kwargs):
    "get a Tabular+Img model from data"
    tabular_il, features_il = data.item_lists
    emb_szs = tabular_il.get_emb_szs(ifnone(emb_szs, {}))
    n_features = features_il[0].data.shape[-1]
    model = TabularFeatureModel(n_features, emb_szs, len(tabular_il.cont_names), out_sz=data.c, 
                              layers=layers, ps=ps, emb_drop=emb_drop, y_range=y_range, use_bn=use_bn)
    return Learner(data, model, **learn_kwargs)

class TabularImageModel(TabularModel):
    def __init__(self, img_model = models.xresnet34, *args, **kwargs):
        body = get_features_map(img_model())
        self.n_features = num_features_model(body)
        super().__init__(*args, **kwargs)
        self.img_model = nn.Sequential(body, nn.AdaptiveAvgPool2d(1), Flatten())
        self.bn_img = nn.BatchNorm1d(self.n_features)
        
    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont + self.n_features] + layers + [out_sz]
    
    def forward(self, x_tabular:Tensor, x_image:Tensor) -> Tensor:
        x_cat, x_cont = x_tabular
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        if self.n_features != 0:
            x_image = self.bn_img(self.img_model(x_image))
            x = torch.cat([x, x_image], 1)
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x

def tabular_image_learner(data:DataBunch, 
                          layers:Collection[int], 
                          emb_szs:Dict[str,int]=None, 
                          ps:Collection[float]=None, 
                          emb_drop:float=0., 
                          y_range:OptRange=None, 
                          use_bn:bool=True, 
                          img_arch=models.xresnet34,
                          **learn_kwargs):
    "get a Tabular+Img model from data"
    tabular_il = data.item_lists[0]
    emb_szs = tabular_il.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularImageModel(img_arch, emb_szs, len(tabular_il.cont_names), out_sz=data.c, 
                              layers=layers, ps=ps, emb_drop=emb_drop, y_range=y_range, use_bn=use_bn)
    return Learner(data, model, **learn_kwargs)

