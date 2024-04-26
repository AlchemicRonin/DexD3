# CHZ
## TODO:
- [x] bimanual RL env step
- [ ] pretain pipeline
  - [x] base rl env
  - [ ] task env
    - [x] laptop
  - [x] generate data
    - [x] segmentation
    - [x] visualization
    - [ ] imagination
## log
- 0421:
  - altas laptop env
    - run: `python examples/random_action.py --task_name=laptop`
    - main change: 
      - dexart/env/rl_env/base.py
      - dexart/env/rl_env/laptop_env.py
      - dexart/env/rl_env/pc_processing.py
      - dexart/env/sim_env/laptop_env.py
  - pretain generate data
    - [ ] imagination is not finished yet
    - run: `python examples/pretrain/generate_dataset.py --task_name=laptop`
      - **you may set `use_gui=False` to speed up**
      - **you may set smaller `n_fold` (=1) to speed up**
    - run: `python examples/pretrain/segmentation/data_utils.py`
      - show the result of generated data
- 0406: 
  - bimanual RL env step done
  - run: `python examples/random_action.py --task_name=laptop`
  - main change: dexart/env/rl_env/bibase.py


# ZJW
- to use tensorboard, first downgrade numpy to 1.21
```
conda uninstall numpy

conda install numpy==1.21
```
- then install tensorboard
```
conda install tensorboard==2.10.0
```
- start train the policy

```
python examples/train.py --pretrain_path log/pn_100.pth --freeze
```
- visuallize trained policy
```
python examples/visualize_policy.py --task_name laptop --checkpoint_path ./examples/model_300.zip
```

# YG