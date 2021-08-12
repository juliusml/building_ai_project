# Early experimentation

from transvec.transformers import TranslationWordVectorizer as TWV
import gensim.downloader as gsd

def model_import():
    """Imports all needed models."""
    # "load" only downloads it if it isn't already on the PC locally
    en_model = gsd.load('en§§') 
    se_model = gsd.load('se§§') 

en_se_model = TWV(en_model, se_model).fit(train)
