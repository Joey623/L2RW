# From Laboratory to Real World: A New Benchmark Towards Privacy-Preserved Visible-Infrared Person Re-Identification
Welcome to use the code from our paper "From Laboratory to Real World: A New Benchmark Towards Privacy-Preserved Visible-Infrared Person Re-Identification". 

## Requirement
```
cuda >= 11.7
torch >= 2.0.1
```

## Dataset
The data structures are as follows.
```
L2RW
 |----data
       |----SYSU-MM01
       |----RegDB
       |----LLCM
 |----CI
 |----EI
 |----ES
```

## Training and Testing

1. **Preprocess the dataset.**

   For each protocol, run the `process.py(EI & ES)` or `pre_process_sysu.py, pre_process_regbd.py, pre_process_llcm.py(CI)`.

2. **Start training**

   Run the `train.py`, and you can also find training details in the `main.ipynb` or `train.ipynb` .

3. **Test**

   Just run the `test.py` or `test.ipynb`

## Acknowledgement

Some methods under ES protocol are implemented according to the official project, which can be found in [AGW](https://github.com/mangye16/ReID-Survey), [CAJ](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ), [Lba](https://github.com/cvlab-yonsei/LbA), [DEEN](https://github.com/ZYK100/LLCM). We thanks the authors for making those repositories public.

The DNS is my another work which published in ECCV24, you can find the [paper](https://fq.pkwyx.com/default/https/www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09119.pdf) and [code](https://github.com/Joey623/DNS).

## Contact

If you have any questions, please feel free to open an issue or contact me via [yan.jiang@oulu.fi](yan.jiang@oulu.fi).

