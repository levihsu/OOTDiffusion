# Self Correction for Human Parsing

We propose a simple yet effective multiple human parsing framework by extending our self-correction network.

Here we show an example usage jupyter notebook in [demo.ipynb](./demo.ipynb).

## Requirements

Please see [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for further requirements.

## Citation

Please cite our work if you find this repo useful in your research.

```latex
@article{li2019self,
  title={Self-Correction for Human Parsing},
  author={Li, Peike and Xu, Yunqiu and Wei, Yunchao and Yang, Yi},
  journal={arXiv preprint arXiv:1910.09777},
  year={2019}
}
```

## Visualization

* Source Image.
![demo](./demo/demo.jpg)
* Instance Human Mask.
![demo-lip](./demo/demo_instance_human_mask.png)
* Global Human Parsing Result.
![demo-lip](./demo/demo_global_human_parsing.png)
* Multiple Human Parsing Result.
![demo-lip](./demo/demo_multiple_human_parsing.png)

## Related

Our implementation is based on the [Detectron2](https://github.com/facebookresearch/detectron2).

