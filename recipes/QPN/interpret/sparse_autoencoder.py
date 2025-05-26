
class MaskTemp:
    """Keeps track of temperature for masking

    Arguments
    ---------
    steps: int
        Number of steps to reduce temperature over
    start_temp: float
        Starting temperature, usually > 1.0
    stop_temp: float
        Final temperature, usually between 0. and 1.
    """

    def __init__(self, steps, start_temp=2.0, stop_temp=0.1):
        assert start_temp > stop_temp
        assert steps > 0
        self.steps = steps
        self.stop_temp = stop_temp
        self.temperature = start_temp
        self.temperature_step_size = (start_temp - stop_temp) / steps

    def update_temp(self):
        """Update temperature one step while ensuring we don't go below stop_temp"""
        if self.temperature > self.stop_temp:
            self.temperature = max(
                self.temperature - self.temperature_step_size, self.stop_temp
            )

    def __call__(self):
        return self.temperature


class SparseAutoEncoder(nn.Module):
    """Adds a sparse auto-encoder (SAE) layer after a pretrained module.

    Majority of initialization code comes from a reference implementation.
    Github repo: saprmarks/dictionary_learning
    Relevant file: dictionary_learning/dictionary.py

    Arguments
    ---------
    target_module: torch.nn.Module
        A pretrained module for inserting the SAE layer.
    dict_size: int
        The number of neurons in the SAE layer, usually larger than the module output.
    module_out_size: int, optional
        This class can automatically determine the size of the module output only
        in the case where it is a simple linear layer, otherwise, this must be
        specified as an argument to the class.
    activation_fn: torch.nn.Module, default torch.nn.ReLU
        The class to use for activation, usually a ReLU-family function
        that creates sparse outputs by zeroing some outputs.
        Compatible with speechbrain.nnet.activations.JumpReLU().
        Also accepts "mask" which adds mask estimation parameters and loss.
    fidelity_loss_fn: loss fn, default speechbrain.nnet.losses.mse_loss
        A loss function that takes predicions, targets for autoencoder loss.
    storing_activations: bool, default False
        Whether to start storing activations and losses. Can be changed later with
        `enable_storage()` and `disable_storage()`.
    sparse_loss_fn: str, default "L1"
        Options are "L1", "L0", but ensure activation supports the argument
        "sparse_loss" if you would like to compute the "L0" loss.
    mask_temperature: MaskTemp
        The starting, stopping, and number of decay steps for the temperature
        used in the sigmoid for each mask item.


    Example
    -------
    >>> test_input = torch.rand(10)
    >>> module = torch.nn.Linear(10, 20)
    >>> sae = SparseAutoEncoder(module, dict_size=100)
    >>> sae.enable_storage()
    >>> sae(test_input).size()
    torch.Size([20])
    >>> sae.sparse_loss.size()
    torch.Size([100])
    >>> torch.allclose(sae.encode(test_input), sae.get_activations())
    True
    """

    def __init__(
        self,
        target_module,
        dict_size,
        module_out_size=None,
        activation_fn=torch.nn.ReLU(),
        fidelity_loss_fn=mse_loss,
        storing_activations=False,
        sparse_loss_fn="L1",
        mask_temperature=MaskTemp(start_temp=2.0, stop_temp=0.1, steps=1000),
    ):
        super().__init__()

        self.fidelity_loss_fn = fidelity_loss_fn

        # Ensure module is frozen and collect info
        self.pretrained_module = target_module
        for param in target_module.parameters():
            param.requires_grad = False
            device = param.device
        if module_out_size is None:
            module_out_size = target_module.weight.data.shape[0]

        # Initialize the machinery for caching the activations and loss
        self.fidelity_loss = None
        self.sparse_loss = None
        self.sparse_activations = None
        self.storing_activations = storing_activations
        self.activation_fn = activation_fn
        self.sparse_loss_fn = sparse_loss_fn
        self.mask_temperature = mask_temperature

        # Initialize the parameters according to reference
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(
            torch.empty(module_out_size, dict_size, device=device)
        )
        self.b_enc = nn.Parameter(torch.zeros(dict_size, device=device))

        if self.activation_fn == "mask":
            self.W_mask = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(module_out_size, dict_size, device=device)
                )
            )
            self.b_mask = nn.Parameter(torch.zeros(dict_size, device=device))

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(dict_size, module_out_size, device=device)
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(module_out_size, device=device))
        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

    def forward(self, x):
        """Full forward pass, encode then decode for training."""

        activations, sparse_loss = self.encode(x, sparse_loss=True)
        predictions = self.decode(activations)

        if self.storing_activations:
            self.sparse_activations = activations
            self.sparse_loss = sparse_loss
            targets = self.pretrained_module(x).detach()
            self.fidelity_loss = self.fidelity_loss_fn(predictions, targets)

        return targets #predictions

    def encode(self, x, sparse_loss=False):
        """Returns the sparse dictionary activations used for interpretability.
        
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the adapter model.
        sparse_loss: bool
            Whether to additionally return the sparse loss. In the case of
            JumpReLU this is L0 defined in the module itself, otherwise
            returns L1 loss.

        Returns
        -------
        activations: torch.Tensor
            The dictionary activations.
        sparse_loss: torch.Tensor (conditional)
            The sparsity loss defined by activation or L1.
        """
        if hasattr(self.pretrained_module, "attn_pooling_w"):
            out = self.pretrained_module.attn_pooling_w(x).squeeze(-1).float()
            out = torch.nn.functional.softmax(out, dim=-1).unsqueeze(-1)
            self.attention_scores = out
        if hasattr(self.pretrained_module, "attn_w"):
            self.attention_scores = self.pretrained_module.attn_w(x).softmax(
                dim=1
            )

        inputs = self.pretrained_module(x)
        pre_activations = inputs @ self.W_enc + self.b_enc
        self.pre_activations = pre_activations

        if self.activation_fn == "mask":
            if self.training:
                t = 1 / self.mask_temperature()
                mask = (t * (inputs @ self.W_mask + self.b_mask)).sigmoid()
                # self.diversity_loss = mask.mean(dim=0) * mask.mean(dim=0).log()
                self.diversity_loss = torch.relu(mask.mean(dim=0) - 0.5)
            else:
                mask = (inputs @ self.W_mask + self.b_mask) > 0
            activations = mask.float() * pre_activations
            if sparse_loss:
                loss = (mask.float().mean(dim=0) * self.W_dec.norm(dim=1)).mean()
                return activations, loss
            return activations
        # If the L0 loss is not supported by the activation_fn, this will fail
        elif sparse_loss and self.sparse_loss_fn == "L0":
            return self.activation_fn(pre_activations, sparse_loss=sparse_loss)
        else:
            activations = self.activation_fn(pre_activations)
            if sparse_loss:
                sparse_loss_term = activations.abs().mean(dim=0)
                sparse_loss_term *= self.W_dec.norm(dim=1)
                return activations, sparse_loss_term.mean()
            return activations

    def decode(self, encoding):
        """Returns the decoded activations for use by the next layer."""
        return encoding @ self.W_dec + self.b_dec

    def update_temperature(self):
        """Forward mask temp update to correct class."""
        self.mask_temperature.update_temp()

    def enable_storage(self):
        """Turn on the storage of activations during the forward pass."""
        self.storing_activations = True

    def disable_storage(self):
        """Turn off the storage of activations during the forward pass."""
        self.storing_activations = False

    def get_activations(self, pre_activations=False):
        """Return the stored activations from the most recent forward pass."""
        return self.sparse_activations

    def get_sparse_loss(self):
        """Return the stored sparse loss from the most recent forward pass."""
        return self.sparse_loss

    def get_fidelity_loss(self):
        """Return the stored fidelity loss from the most recent forward pass."""
        return self.fidelity_loss
