from cape_core.classifier import *

@call_parse
def main(folder:Param("Folder of Images. (default: zoom20)", str)='zoom20',
         epochs:Param("Number of epochs.(default: 1)", int)=1,
         lr:Param("Learning rate.(default: 1e-3)", float)=1e-3,
         bs:Param("Batch size. (default: 128)", int)=128,
         mixup:Param("Use Mixup", bool)=True, 
         filename:Param("output filename", str)='features',
         ):
    path = Path('/home/tc256760/Documents/Tabular_image_model/Sample Models/') 
    features = train(path=path,  folder=folder, lr=lr, epochs=epochs, bs=bs, mixup=mixup)
    features.to_pickle(filename)
    print(f'Features saved on :{path/filename}')