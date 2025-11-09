# Inference

## 0.Prepare
- in `/scripts/inference.py`, add
  ```python
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9" # custom by your device
  ```
- in case you get this ImportError:
  ```bash
    ImportError: /root/miniconda3/envs/e2style/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /root/.cache/torch_extensions/py38_cu121/fused/fused.so)
  ```
    execute this:
    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```
## 1.Inference
- recommend execute all command in global path
- in [parameter_check.py](debug/parameter_check.py), you can find detailed parameter analyse
- in `models/face_frontend.py`, set `self.encoder.load_state_dict(strict=False)`, in case there are more parameters in open source weight `multi.pt`
- for single img tasks
  - align all input img first
  ```bash
  python scripts/align_all_parallel.py --root_path=test_datasets/test_single
  ```
  - then start inference
  ```bash
  python scripts/train.py
  ```

## 2.supplement
- supplementary datasets: 
  - [multi-pie](https://www.kaggle.com/datasets/aliates/multi-pie)
  - [FFWM](https://github.com/csyxwei/FFWM)