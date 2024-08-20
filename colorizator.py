import torch
from torchvision.transforms import ToTensor
import numpy as np

from networks.models import Colorizer
from denoising.denoiser import FFDNetDenoiser
from utils.utils import resize_pad

class MangaColorizator:
    def __init__(self, device, generator_path='networks/generator.zip', extractor_path='networks/extractor.pth'):
        self.device = device

        # Load the colorizer model and move it to the correct device
        self.colorizer = Colorizer().to(self.device)
        self.colorizer.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.colorizer = self.colorizer.eval()

        # Load the denoiser and move it to the correct device
        self.denoiser = FFDNetDenoiser(_device=self.device)

        self.current_image = None
        self.current_hint = None
        self.current_pad = None

    def set_image(self, image, size=576, apply_denoise=True, denoise_sigma=25, transform=ToTensor()):
        if size % 32 != 0:
            raise RuntimeError("size is not divisible by 32")

        # Apply denoising if requested and move image to GPU
        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma=denoise_sigma)

        # Resize and pad the image, move it to GPU, and apply transformation
        image, self.current_pad = resize_pad(image, size)
        self.current_image = transform(image).unsqueeze(0).to(self.device)

        # Initialize current hint as a tensor on the GPU
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2], self.current_image.shape[3]).float().to(self.device)

    def update_hint(self, hint, mask):
        # Convert hint and mask to appropriate formats and move them to GPU
        if issubclass(hint.dtype.type, np.integer):
            hint = hint.astype('float32') / 255

        hint = (hint - 0.5) / 0.5
        hint = torch.FloatTensor(hint).permute(2, 0, 1).to(self.device)
        mask = torch.FloatTensor(np.expand_dims(mask, 0)).to(self.device)

        # Update the hint with new data, ensuring it's moved to the GPU
        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

    def colorize(self):
        with torch.no_grad():
            # Perform colorization on the GPU
            fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
            fake_color = fake_color.detach()

        # Move the result back to CPU for further processing
        result = fake_color[0].cpu().permute(1, 2, 0) * 0.5 + 0.5

        # Adjust for padding if necessary
        if self.current_pad[0] != 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] != 0:
            result = result[:, :-self.current_pad[1]]

        return result.numpy()
