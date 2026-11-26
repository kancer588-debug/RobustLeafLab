# Third-party sources

## PlantCity v4 sample images

The nine photographs in `examples/images/` come from the PlantCity v4 publisher test archive. Muhammad Sheraz Khan, Kainat Nisa, and Irshad Ahmad are the contributors named in the dataset record. The files were renamed for the example and kept byte-for-byte. `examples/manifest.csv` records each archive member and its SHA-256. The class labels are the publisher's labels, and the reference predictions are from RobustLeafLab.

- Dataset and attribution: [PlantCity v4, DOI 10.17632/w8kh2xkspx.4](https://data.mendeley.com/datasets/w8kh2xkspx/4)
- Image license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The full PlantCity replay manifest contains file identifiers, labels, and predictions. It refers to the publisher archive for the remaining image bytes.

## DINOv2

The inference program downloads the upstream general-purpose DINOv2 ViT-S/14 code and pretrained weights. The fixed upstream revision, weight URL, and SHA-256 values are in [backbone.py](robustleaflab/backbone.py). Meta releases this upstream model (code and weights) under the Apache License 2.0. The published RobustLeafLab core ZIP contains the project baseline and classifier heads.

- [Meta DINOv2 pinned documentation](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md#license)
- [Meta DINOv2 license at that revision](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
