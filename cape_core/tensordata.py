from .imports import *
from .utils import *

__all__ = ['TensorList', 'tensor_lists_from_arrays']

class TensorItem(ItemBase):
    "`ItemBase` suitable for Tensors"

    def __init__(self, item):
        super().__init__(item)
        self.channels = item.shape[-2]
        self.seq_len = item.shape[-1]


    def __str__(self):
        return 'TensorItem(ch={:.0f}, seq_len={:.0f})'.format(
            self.channels, self.seq_len)

    def clone(self):
        return self.__class__(self.data.clone())

    def apply_tfms(self, tfms, **kwargs):
        x = self.clone()
        for tfm in tfms:
            x.data = tfm(x.data)
        return x

    def reconstruct(self, item):
        return TensorItem(item)

    def show(self, ax=None, title=None, **kwargs):
        if ax is None:
            plt.plot(*self.data)
            plt.show()
        else:
            ax.plot(*self.data)
            ax.title.set_text(title)
            ax.tick_params(
                axis='both',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off',
                right='off',
                left='off',
                labelleft='off')
            return ax

class TensorPreProc(PreProcessor):

    def __init__(self, ds: ItemList): self.ds = ds

    def process(self, ds: ItemList):
        ds.features, ds.seq_len = self.ds.get(0).data.size(-2), self.ds.get(0).data.size(-1)


class TensorList(ItemList):
    "`ItemList` suitable for Tensor"
    _processor = TensorPreProc
    _label_cls = None
    _square_show = True

    def __init__(self, items, *args, mask=None, tfms=None, **kwargs):
        items = to3dtensor(items)
        super().__init__(items, *args, **kwargs)
        self.tfms,self.mask = tfms,mask
        self.copy_new.append('tfms')

    def get(self, i):
        item = super().get(i)
        if self.mask is None: return TensorItem(to2dtensor(item))
        else: return[TensorItem(to2dtensor(item[m])) for m in self.mask]


    def show_xys(self, xs, ys, figsize=(10, 10), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows, rows, figsize=figsize)
        for x, y, ax in zip(xs, ys, axs.flatten()):
            with np.printoptions(precision=3, suppress=True):  #print only 3 decimals in title
                x.show(ax=ax, title=str(y), **kwargs)
        # plt.tight_layout()
        plt.show()

    def show_xyzs(self, xs, ys, zs, figsize=(10, 10), **kwargs):
        if self._square_show_res:
            rows = int(np.ceil(math.sqrt(len(xs))))
            fig, axs = plt.subplots(
                rows,
                rows,
                figsize=figsize)
            for x, y, z, ax in zip(xs, ys, zs, axs.flatten()):
                x.show(ax=ax, title=f'{str(y)}\n{str(z)}', **kwargs)
        else:
            fig, axs = plt.subplots(
                len(xs),
                2,
                figsize=figsize)
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                x.show(ax=axs[i, 0], title=str(y), **kwargs)
                ##TODO replace this curve with the computed one from pvlib
                x.show(ax=axs[i, 1], title=str(z), **kwargs)
        # plt.tight_layout()
        plt.show()

    @classmethod
    def from_array(cls, ts, **kwargs):
        return cls(ts)

    @classmethod
    def from_df(cls, df, path='.', cols=None, feat=None, processor=None, **kwargs) -> 'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `df`."
        if cols is 0:
            inputs = df
        else:
            col_idxs = df_names_to_idx(list(cols), df)
            inputs = df.iloc[:, col_idxs]
        assert inputs.isna().sum().sum() == 0, f"NaN values in column(s) {cols} of your dataframe, fix it."
        inputs = df2array(inputs, feat)
        res = cls(
            items=inputs,
            path=path,
            inner_df=df,
            processor=processor,
            **kwargs)
        return res

def tensor_lists_from_arrays(X_train, y_train, X_valid, y_valid, label_cls=FloatList):
    src = ItemLists('.', TensorList(X_train), TensorList(X_valid))
    return src.label_from_lists(y_train, y_valid, label_cls=FloatList)