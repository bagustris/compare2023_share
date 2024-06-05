import torch
import torch.nn as nn
from torchviz import make_dot

class Params:
    def __init__(self):
        self.pool = "attention"
        self.embedding_size = 256
        self.num_heads = 8
        self.num_layers = 2

class StackedModule(nn.Module):
    """
    Stack model which sends the inputs through multiple heads, concatenating features with the output from the lower stages.
    """

    def __init__(self, feat_dim:int, params:Params) -> None:
        super().__init__()
        self.params = params


        # attention pooling of the features for each task
        if self.params.pool == "attention":
            self.pools = nn.Linear(feat_dim, 1, bias=False)
        else: # avg pool 
            self.pools = None

        embedding_size = params.embedding_size

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

        input_feat = inputs # update this variable across the chain
        high_pred = self.high_encoder(input_feat)
  
        return {"high": high_pred}

# create dummy data to plot the model
feat_dim = 1024
seqlen = 10
batch_size = 8 

# create dummy inputs and batch
dummy_inputs = torch.randn(batch_size, seqlen, feat_dim)
dummy_batch = {}

# create model instance
params = Params()
model = StackedModule(feat_dim, params)

# forward pass with dummy data
outputs = model(dummy_inputs, dummy_batch)

# plot the model using torchviz
dot = make_dot(outputs["high"], params=dict(model.named_parameters()),
               show_attrs=True, show_saved=True)
dot.render("stacked_module_plot", format="pdf")
