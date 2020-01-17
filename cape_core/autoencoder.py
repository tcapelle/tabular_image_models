def print_shape(x):
    print(x.shape)
    return x

class UpSample(nn.Module):
    def __init__(self,feat_in,feat_out,out_shape=None,scale=2):
        super().__init__()
        self.conv = nn.Conv2d(feat_in,feat_out,kernel_size=(3,3),stride=1,padding=1)
        self.out_shape,self.scale = out_shape,scale  
    def forward(self,x):
        return self.conv(
            F.interpolate(
                x,size=self.out_shape,scale_factor=self.scale,mode='bilinear',align_corners=True))
    
def upsample_layer(feat_in, feat_out, out_shape=None, scale=2, act='relu'):    
    upSamp = UpSample(feat_in,feat_out,out_shape=out_shape,scale=scale)    
    layer = nn.Sequential(upSamp, nn.ReLU(inplace=True), nn.BatchNorm2d(feat_out))
    return layer

def simple_encoder(filters=[3, 8, 16, 32, 64], debug=False):
    features = list(zip( filters[:-1], filters[1:]))
    layers = []
    if debug: layers.append(Lambda(print_shape))
    for ni, no in features:
        norm_type = None if no==filters[-1] else NormType.Batch
        layers.append(conv_layer(ni, no, ks=3, stride=2, padding=1, norm_type=norm_type))
        layers.append(conv_layer(no, no, ks=3, stride=1, padding=1, norm_type=norm_type))
        if debug: layers.append(Lambda(print_shape))
    return nn.Sequential(*layers)

def simple_decoder(filters=[3, 8, 16, 32, 64], y_range=(-3,3), debug=False):
    features = list(zip(filters[-1:0:-1], filters[-2:-len(filters)-1:-1]))
    layers = []
    if debug: layers.append(Lambda(print_shape))
    for ni, no in features:
        layers.append(upsample_layer(ni,no))
        if debug: layers.append(Lambda(print_shape))
    return nn.Sequential(*layers, SigmoidRange(*y_range))