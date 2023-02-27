import os

# update these paths to point to your FCE and W&I+LOCNESS downloads
FCE_SOURCE = '/path/to/downloaded/fce/m2'
WIBEA_SOURCE = '/path/to/downloaded/wi+locness/m2'

# where to save the datasets
TARGET = 'datasets/wibea'

class PathInfo(object):
    def __init__(self, source, m2, conll):
        self.source = source
        self.m2 = m2
        self.conll = conll


fcetrain = PathInfo(
    source=os.path.join(FCE_SOURCE, 'fce.train.gold.bea19.m2'),
    m2=os.path.join(TARGET, 'fce_m2/fce.train.gold.rverbsva.m2'),
    conll=os.path.join(TARGET, 'fce_conll/fce.train.gold.rverbsva.conll')
)

wibea_train = PathInfo(
    source=os.path.join(WIBEA_SOURCE, 'ABC.train.gold.bea19.m2'),
    m2=os.path.join(TARGET, 'wibea_m2/wibea.train.gold.rverbsva.m2'),
    conll=os.path.join(TARGET, 'wibea_conll/wibea.train.gold.rverbsva.conll')
)

wibea_dev = PathInfo(
    source=os.path.join(WIBEA_SOURCE, 'ABCN.dev.gold.bea19.m2'),
    m2=os.path.join(TARGET, 'wibea_m2/wibea.dev.gold.rverbsva.m2'),
    conll=os.path.join(TARGET, 'wibea_conll/wibea.dev.gold.rverbsva.conll')
)

data = {
    'fce-train': fcetrain,
    'wibea-train': wibea_train,
    'wibea-dev': wibea_dev
}