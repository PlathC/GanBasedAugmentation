"""
    This file contains pair generator based on https://www.justinpinkney.com/stylegan-network-blending/
    Not used in the paper results
"""

import torch
from typing import Dict, List, Tuple

from .stylegan2 import StyleGenerator


def stylegan_partial_forward(ws: torch.Tensor,
                             generator: StyleGenerator,
                             block_resolutions: List[int],
                             synthesis_kwargs: Dict,
                             x: torch.Tensor = None,
                             img: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

    block_ws = []
    ws = ws.to(torch.float32)
    w_idx = 0
    for res in block_resolutions:
        block = getattr(generator.synthesis, f'b{res}')
        block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
        w_idx += block.num_conv

    for res, cur_ws in zip(block_resolutions, block_ws):
        block = getattr(generator.synthesis, f'b{res}')
        x, img = block(x, img, cur_ws, **synthesis_kwargs)

    return x, img


def stylegan_mixing(g1: StyleGenerator,
                    g2: StyleGenerator,
                    z: torch.Tensor,
                    mixing_lvl: int) -> torch.Tensor:
    synthesis_kwargs = {
        'noise_mode': 'const',
        'force_fp32': True
    }

    ws = g1.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8)
    x, img = stylegan_partial_forward(
        ws,
        g1,
        g1.synthesis.block_resolutions[0:mixing_lvl],
        synthesis_kwargs
    )

    ws = g2.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8)
    _, img = stylegan_partial_forward(
        ws,
        g2,
        g2.synthesis.block_resolutions[mixing_lvl:],
        synthesis_kwargs,
        x=x,
        img=img
    )

    return img


class PairsStyleGenerator:
    def __init__(self,
                 content_checkpoint: str,
                 style_checkpoint: str,
                 device: torch.device = torch.device('cuda:0')):
        self.content_generator = StyleGenerator(content_checkpoint).to(device)
        self.style_generator = StyleGenerator(style_checkpoint).to(device)

    def generate_content_image(self,
                               noise: torch.Tensor = None,
                               device: torch.device = torch.device('cuda:0')) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = self.content_generator.produce_noise(1, device=device)
        return self.content_generator(noise), noise

    def generate_mixed_image(self,
                             noise: torch.Tensor,
                             mixing_level: int) -> torch.Tensor:
        return stylegan_mixing(
            self.content_generator.G,
            self.style_generator.G,
            noise,
            mixing_level
        )


class PairsManagerGenerator(PairsStyleGenerator):
    def __init__(self,
                 content_checkpoint: str,
                 style_checkpoint: str,
                 image_number: int,
                 device: torch.device = torch.device('cuda:0')) -> None:
        super().__init__(content_checkpoint, style_checkpoint, device)

        self.device = device
        self.image_number = image_number
        self.noise = []
        for i in range(self.image_number):
            self.noise += [self.content_generator.produce_noise(1, torch.device('cpu'))]

    def content_image_by_index(self, index: int) -> torch.Tensor:
        assert index < self.image_number, 'index should be less than {}'.format(self.image_number)
        return self.content_generator(self.noise[index].to(self.device))

    def styled_image_by_index(self, index: int) -> torch.Tensor:
        assert index < self.image_number, 'index should be less than {}'.format(self.image_number)
        return self.style_generator(self.noise[index].to(self.device))

    def mixed_image_by_index(self, index: int, mixed_level: int) -> torch.Tensor:
        assert index < self.image_number, 'index should be less than {}'.format(self.image_number)
        return self.generate_mixed_image(self.noise[index].to(self.device), mixed_level)
