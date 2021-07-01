import pickle
import torch

import dnnlib


class StyleGenerator(torch.nn.Module):
    """
        StyleGAN2 based generator, based on https://github.com/NVlabs/stylegan2-ada-pytorch
    """
    def __init__(self, checkpoint: str) -> None:
        """
        Load the StyleGAN2 weights
        Args:
            checkpoint: Checkpoint location
        """
        super().__init__()
        with dnnlib.util.open_url(checkpoint) as f:
            self.G = pickle.load(f)['G_ema']  # type: ignore

    def produce_noise(self, batch_size, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Produce noise for this generator
        Args:
            batch_size: Number of batch to generate
            device: Device of the output tensor

        Returns:
            Created noise tensor
        """
        return torch.randn([batch_size, self.G.z_dim]).to(device)

    def forward(self, z: torch.Tensor, c: int = None) -> torch.Tensor:
        """
        Pass forward of StyleGAN2
        Args:
            z: Noise tensor
            c: class labels

        Returns:
            Artificial image
        """
        return self.G(z, c)
