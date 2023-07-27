import enum
from sqlite3 import paramstyle
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from end2you.utils import Params

# from model import WAV2VEC2_BASE_PATH, get_feature_dim
from model import get_feature_dim
import dataset

"""
Model definitions
"""

def count_all_parameters(model:torch.nn.Module) -> int:

    return sum([p.numel() for p in model.parameters()])


def count_trainable_parameters(model:torch.nn.Module) -> int:

    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def load_ssl_model(params:Params) -> nn.Module:
    """
    Loads a SSL Transformer model
    :params model Params object
    """

    ssl_model = params.feature_extractor
    #SpecAugment Args
    mask_time_prob = params.augment.mask_time_prob
    mask_time_length = params.augment.mask_time_length
    mask_feature_prob = params.augment.mask_feature_prob
    mask_feature_length = params.augment.mask_feature_length

    # list of model form facebook
    model_fb = ["wav2vec2-base", "wav2vec2-xls-r-2b", "wav2vec2-xls-r-1b",  
                "wav2vec2-large", "wav2vec2-xls-r-300m", "hubert-base-ls960", 
                "hubert-large-ll60k"]
    model_ms =["wavlm-base-plus", "wavlm-large", "unispeech-sat-plus", "unispeech-sat-large"]


    print("SSL Model: ", ssl_model)

    if ssl_model in model_fb:
        # load pre-trained model from facebook
        model = transformers.Wav2Vec2Model.from_pretrained(
            f"facebook/{ssl_model}",
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            mask_feature_prob=mask_feature_prob,
            mask_feature_length=mask_feature_length
        )

    elif ssl_model in model_ms:
        #model = transformers.Wav2Vec2Model.from_pretrained(WAV2VEC2_BASE_PATH,
        model = transformers.Wav2Vec2Model.from_pretrained(
            f"microsoft/{ssl_model}",
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            mask_feature_prob=mask_feature_prob,
            mask_feature_length=mask_feature_length
        )
    
    # audeering model
    elif ssl_model == 'audeering':
        model = transformers.Wav2Vec2Model.from_pretrained(f"audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            mask_feature_prob=mask_feature_prob,
            mask_feature_length=mask_feature_length
        )
    else:
        raise NotImplementedError

    return model

class BaseMultiModule(nn.Module):
    """
    Baseline Model which routes the features from the extractor through independent shallow networks
    """
    def __init__(self, feat_dim:int, params:Params) -> None:
        super().__init__()
        self.params = params
        self.is_training = False

        # attention pooling of the features for each task
        if self.params.pool == "attention":
            self.pools = nn.ModuleList([nn.Linear(feat_dim, 1, bias=False) for i in range(params.num_outputs)])
        else: # avg pool 
            self.pools = []

        embedding_size = params.embedding_size

        # # output heads - 2 layer networks
        # self.voc_type_model = nn.Sequential(
        #     nn.Linear(feat_dim, embedding_size),
        #     nn.BatchNorm1d(embedding_size),
        #     nn.GELU(),
        #     nn.Linear(embedding_size, 9),
        #     nn.BatchNorm1d(9),
        # )

        self.high_model = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            # nn.InstanceNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 9),
            nn.BatchNorm1d(9),
            # nn.InstanceNorm1d(9),
            nn.Sigmoid()
        )


    def forward(self, inputs, batch=None):
        """
        inputs: [B, seqlen, featdim]
        """
        # first pool the features over time
        if self.params.pool == "attention":
            # attention per output
            # voc_type_feat = torch.sum(torch.softmax(self.pools[0](inputs), dim=1) * inputs, dim=1)
            high_feat = torch.sum(torch.softmax(self.pools[2](inputs), dim=1) * inputs, dim=1)
            # pass

            # out_voc = self.voc_type_model(voc_type_feat)
            out_high = self.high_model(high_feat)


        else: # avg pool
            inputs = torch.mean(inputs, dim=1)
            # each head gets the same averaged features
            # out_voc = self.voc_type_model(inputs)
            out_high = self.high_model(inputs)

        # return a dict per task
        return {"high": out_high}


class StackedModule(nn.Module):
    """
    Stack model which sends the inputs through multiple heads, concatenating features with the output from the lower stages.
    It goes in a fixed order type -> low -> high -> culture
    """

    def __init__(self, feat_dim:int, params:Params) -> None:
        super().__init__()
        self.params = params

        # is training flag to switch chain from GT to predictions 
        # self.is_training = False  not needed since already in nn.Module

        # attention pooling of the features for each task
        if self.params.pool == "attention":
            #self.pools = nn.ModuleList([nn.Linear(feat_dim, 1, bias=False) for i in range(params.num_tasks)])
            self.pools = nn.Linear(feat_dim, 1, bias=False)
        else: # avg pool 
            #self.pools = []
            self.pools = None

        embedding_size = params.embedding_size

        # encoders
        # self.voc_type_encoder = nn.Sequential(
        #     nn.Linear(feat_dim, embedding_size),
        #     nn.BatchNorm1d(embedding_size),
        #     nn.GELU(),
        #     nn.Linear(embedding_size, 9),
        #     nn.BatchNorm1d(9),
        # )

        # # feeds into
        # high_feat_dim = feat_dim + 2
        self.high_encoder = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            # nn.InstanceNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 9),
            nn.BatchNorm1d(9),
            # nn.InstanceNorm1d(9),
            nn.Sigmoid()
        )


    def forward(self, inputs:torch.Tensor, batch:dict):
        """
        inputs: [B, seqlen, feat_dim]
        """

        # first aggregate the features over time
        if self.params.pool == "attention":
            weights = torch.softmax(self.pools(inputs), dim=1)
            inputs = torch.sum(weights * inputs, dim=1)
        else:   # avg pool
            inputs = torch.mean(inputs, 1)
        
        # during training time, the ground truth is fed to the model stages. When evaluating, the encoder outputs are used.

        # voc_type_label = batch.get("voc_type")  # [B, 8]
        # high_label = batch.get("high")  # [B, 10]

        input_feat = inputs # update this variable across the chain

        # Type
        # voc_type_pred = self.voc_type_encoder(inputs)
        # # check if GT exists or we are in prediction mode
        # if voc_type_label is None or not self.training:  # test or val case
        #     # add a softmax to the type predictions 
        #     vl = torch.softmax(voc_type_pred, dim=-1)
        #     high_input_feat = torch.cat([inputs, vl], dim=-1)
        # elif voc_type_label is not None and self.training:   # train case
        #     # convert the voc_type to one hot encoding
        #     vl = F.one_hot(voc_type_label, 8)
        #     high_input_feat = torch.cat([inputs, vl], dim=-1).type_as(inputs)
        # else:   # training with no GT label
        #     raise NotImplementedError


        # High
        high_pred = self.high_encoder(input_feat)
  
        return {"high": high_pred}


class MLP(nn.Module):
    def __init__(self, feat_dim: int, params: Params) -> None:
        super().__init__()
        self.params = params
        self.linear = nn.Linear(feat_dim, 9)

    def forward(self, features, batch):
        pooled = features.mean(dim=1)
        predicted = self.linear(pooled)
        return {"high": predicted}



class AbstractModel(nn.Module):

    """
    base class with feature extractor and classifier modules
    """

    def __init__(self, params:Params) -> None:
        super().__init__()
        self.params = params    # store these here

        self.feature_extractor = load_ssl_model(params)
        self.classifier = nn.Module()

    def forward(self, inputs, batch):
        """
        generic processing of fe features here?
        """
        if self.params.model.features == "attention":
            features =  self.feature_extractor(inputs)
        else:
            features = self.feature_extractor(inputs, return_hidden=True)

        return self.classifier(features)


    def freeze_fe(self):
        """
        Freezes the feature extractor layers
        """
        self.feature_extractor.requires_grad_(False)

    def unfreeze_fe(self):
        """
        Unfreezes the feature extractor layers
        """
        self.feature_extractor.requires_grad_(True)


class StackModel(AbstractModel):
    """
    Stacked (Chained) Model that uses the predictions of one task for the next
    """

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        feat_dim = get_feature_dim(params)
        self.classifier = StackedModule(feat_dim=feat_dim, params=params)

    def forward(self, inputs, batch):
        # feature extraction
        if self.params.features == "cnn": # pick only cnn features
            features = self.feature_extractor(inputs)
            features = features["extract_features"]
        elif self.params.features == "both":
            features = self.feature_extractor(inputs, return_hidden=True)
            features = torch.cat([features["last_hidden_state"], features["extract_features"]], dim=-1)
        else:   # only last layer 
            features = self.feature_extractor(inputs)
            features = features["last_hidden_state"]
        # classifier
        out = self.classifier(features, batch)

        return out


class NN_Model(AbstractModel):
    """Class to wrap a simple NN model"""
    
    def __init__(self, params: Params) -> None:
        super().__init__(params)

        feat_dim = get_feature_dim(params)
        self.classifier = MLP(feat_dim=feat_dim, params=params)

    def forward(self, inputs, batch):
        # feature extraction
        if self.params.features == "cnn": # pick only cnn features
            features = self.feature_extractor(inputs)
            features = features["extract_features"]
        elif self.params.features == "both":
            features = self.feature_extractor(inputs, return_hidden=True)
            features = torch.cat([features["last_hidden_state"], features["extract_features"]], dim=-1)
        else:   # only last layer 
            features = self.feature_extractor(inputs)
            features = features["last_hidden_state"]
        # classifier
        out = self.classifier(features, batch)

        return out


def model_factory(model_params:Params) -> AbstractModel:
    """
    Factory method that builds a model from the catalogue based on the specified params
    :model_params a Params object containing model architecture info
    :returns: A subclass of Abstractmodel
    """

    if model_params.model_name == "stacked":
        return StackModel(model_params)
    elif model_params.model_name == "nn":
        return NN_Model(model_params)
    else:
        raise ValueError("Model architecture {} is not recognised".format(model_params.model_name))



            

        
        


    
