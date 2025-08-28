# L2RW+: A Comprehensive Benchmark Towards Privacy-Preserved Visible-Infrared Person Re-Identification

Welcome to use the code from our paper "L2RW+: A Comprehensive Benchmark Towards Privacy-Preserved Visible-Infrared Person Re-Identification".  This repo provides the implementation of our L2RW+ on video VI-ReID datasets.

## Dataset and Preprocess

* [HITSZ-VCM ](https://github.com/VCM-project233/MITML)

  ````
  set the root='your_data_path' in VCM class in data_manager.py
  ````

* [BUPTCampus](https://github.com/dyhBUPT/BUPTCampus)

  ````
  set the data_root in opts.py
  ````

## Training

````
# CI protocol
# vcm
python train.py --lr 0.2 --seq_lenth 6 --batch-size 8 --max_epoch 100 --num_pos 4 --test-batch 32 --seed 0

# bupt
python train.py --lr 0.2 --train_bs 16 --max_epoch 100 --data_root $LOCAL_SCRATCH/DATA --sequence_length 10 --img_hw 288 144 
````

## Testing

````
# CI protocol
# vcm
python test.py

# bupt
python test.py
````

**How to achieve cross-dataset evaluation (H$$\rightarrow$$B, B$$\rightarrow$$H)?**

Take the H$$\rightarrow$$B as an example:

1. In the bupt project, set the model hyperparameter `class_num` in `test.py` to match the number of classes in the H dataset.
2. Load the model weight that trained on the H dataset, and then run `test.py` in bupt project.

You can follow this to achieve any cross-dataset evaluation in ReID, whether centralized training or decentralized training,  whether single-modality or cross-modality.

## Acknowledge

This project is based on the [HITSZ-VCM ](https://github.com/VCM-project233/MITML) and [BUPTCampus](https://github.com/dyhBUPT/BUPTCampus). We thanks the authors for making those repositories and datasets public.

## Contact

If you have any questions, please feel free to open an issue or contact me via yan.jiang@oulu.fi.

# #Citation

````
@inproceedings{jiang2025laboratory,
  title={From laboratory to real world: A new benchmark towards privacy-preserved visible-infrared person re-identification},
  author={Jiang, Yan and Yu, Hao and Cheng, Xu and Chen, Haoyu and Sun, Zhaodong and Zhao, Guoying},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={8828--8837},
  year={2025}
}

@inproceedings{jiang2024domain,
  title={Domain shifting: A generalized solution for heterogeneous cross-modality person re-identification},
  author={Jiang, Yan and Cheng, Xu and Yu, Hao and Liu, Xingyu and Chen, Haoyu and Zhao, Guoying},
  booktitle={European Conference on Computer Vision},
  pages={289--306},
  year={2024},
  organization={Springer}
}
````



