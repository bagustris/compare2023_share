from pathlib import Path
from end2you.utils import Params

"""
Model module
"""

# WAV2VEC2_BASE_PATH = "/data/eihw-gpu5/karasvin/models/pretrained/facebook/wav2vec2-base/Model"

def get_feature_dim(params:Params)-> int:
    """
    Helper to calculate the number of input features from the feature extractor
    """

    # if any["wav2vec2", "wavlm-base-plus", "unispeech-sat-plus",
    #        "hubert-base-ls960"] in params.feature_extractor:

    if any(feature in params.feature_extractor for feature in ["wav2vec2-base", 
           "wavlm-base-plus", "unispeech-sat-plus", "hubert-base-ls960"]):
        if params.features == "attention":
            return 768
        elif params.features == "cnn":
            return 512
        elif params.features == "both":
            return 768 + 512
    
    if any(feature in params.feature_extractor for feature in ["wav2vec2-large",
           "wavlm-large", "unispeech-sat-large", "hubert-large-ll60k", 
           "wav2vec2-xls-r-300m", "audeering"]):
        if params.features == "attention":
            return 1024

    if params.feature_extractor == "wav2vec2-xls-r-1b":
        return 1280
        
    if params.feature_extractor == "wav2vec2-xls-r-2b":
        if params.features == "attention":
            return 1920
    else:
        raise ValueError



