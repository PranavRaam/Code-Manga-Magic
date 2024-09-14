# Automatic colorization

1. Download [generator](https://drive.google.com/file/d/1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t/view?usp=sharing) and [denoiser](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view?usp=sharing) weights. Put generator and extractor weights in `networks` and denoiser weights in `denoising/models`.
2. To colorize image or folder of images, use the following command:
```
$ python inference.py -p "path to file or folder"
```

| Original      | Colorization      |
|------------|-------------|
| <img src="figures/panel1.jpeg" width="512"> | <img src="figures/panel1_colorized.png" width="512"> |
| <img src="figures/panel2.jpg" width="512"> | <img src="figures/panel2_colorized.png" width="512"> |
| <img src="figures/panel3.jpg" width="512"> | <img src="figures/panel3_colorized.png" width="512"> |
| <img src="figures/panel4.jpg" width="512"> | <img src="figures/panel4_colorized.png" width="512"> |
| <img src="figures/panel5.jpg" width="512"> | <img src="figures/panel5_colorized.png" width="512"> |
| <img src="figures/panel6.jpg" width="512"> | <img src="figures/panel6_colorized.png" width="512"> |
