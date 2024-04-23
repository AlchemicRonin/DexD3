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


# YG