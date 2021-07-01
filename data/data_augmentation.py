import torchvision.transforms.functional as F
import random
from typing import Sequence


class SpecififRotateTransform:
    """
    Helper class to provide a way to set specific rotation angles.
    """
    def __init__(self, angles: Sequence[int]):
        """
        Args:
            angles: The available angles choices
        """
        self.angles = angles

    def __call__(self, x):
        """
        Transform x
        Args:
            x: Image to transform

        Returns:
            A transformed image
        """
        angle = random.choice(self.angles)
        return F.rotate(x, angle)
