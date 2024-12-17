# Implementation of SoundMorpher: Perceptually-Uniform Sound Morphing with Diffusion Model

This is implementation codes of the [paper](https://arxiv.org/pdf/2410.02144): SoundMorpher: Perceptually-Uniform Sound Morphing with Diffusion Model. 

Our demonstration page is in [Demo](https://xinleiniu.github.io/SoundMorpher-demo/).

## Environment setting

This code was tested with Python3.8.10, Pytorch 2.3.0+cu121. SoundMorpher relies on a pretrained [AudioLDM2](https://github.com/haoheliu/AudioLDM2).

You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Quickstart (Jupyter Notebook)

The entire pipeline for SoundMorpher is available at [SoundMorpher.ipynb](SoundMorpher.ipynb).

## Batch execute via command line

To batch execute SoundMorpher, run

```
python SoundMorpher_batch.py --N interpolation number --sampling_steps 100
```

This command require a [meta file](./audio/meta.txt) that stores source and target audio path.

## Evaluation metrics

To obtain evaluation metrics, run

```
python evaluation.py --morph_dir "path to morphed results" --source_dir "path to sourced audios"
```

Morphed results path should have the same structure as the one produced by SounrMorpher_batch.py, i.e., ```morphed_dir/alpha.txt``` and ```morph_dir/audios/morphed_results.wav```

## Citation

If you are interested in our paper, please cite as below

```bibtex
@article{niu2024soundmorpher,
  title={SoundMorpher: Perceptually-Uniform Sound Morphing with Diffusion Model},
  author={Niu, Xinlei and Zhang, Jing and Martin, Charles Patrick},
  journal={arXiv preprint arXiv:2410.02144},
  year={2024}
}
```

## Reference

[IMPUS](https://github.com/GoL2022/IMPUS), [AudioLDM2](https://github.com/haoheliu/AudioLDM2), [AudioLDM-eval](https://github.com/haoheliu/audioldm_eval)
