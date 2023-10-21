# self-supervised-rule-learning
A neuro-symbolic approach to self-learn rules that serve as interpretable knowledge to perform relation linking in knowledge base question answering systems.
## Cluttr Experiments
1. Download [clutrr](https://drive.google.com/file/d/1SEq_e1IVCDDzsBIBhoUQ5pOVH5kxRoZF/view) dataset.
2. Pre-process the data using `python clutrr/preprocess_data.py --input_data_path <clutrr data path> --output_data_path <pre-processed data path>`.
3. Perform rule learning and generate test results by invoking `python clutrr/self_learn_rules.py --train_data_json_path <train json file path> --test_data_json_path <test json file path>`.