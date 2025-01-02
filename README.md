# BROFL

### Description
One-shot federated learning (OFL) is a promising learning paradigm that enables global model training in a single communication round, addressing the efficiency and security challenges posed by multi-round federated learning (FL). However, OFL exhibits an increased susceptibility to Byzantine attacks due to the single communication round. Existing Byzantine-robust FL schemes often struggle to effectively identify malicious clients and mitigate their negative impact on the global model in a single communication round. To address these limitations, we propose a Byzantine-robust OFL (BROFL) framework that integrates hybrid-domain fuzzy clustering with a meta-learning-based correction strategy to enhance the robustness of FL against Byzantine attacks in a single communication round. Specifically, BROFL employs both model-domain and data-domain metrics to identify client discrepancies and distinguishes Byzantine clients from benign ones via hybrid-domain fuzzy clustering. It then applies a meta-learning-based correction mechanism to refine the contributions of identified malicious clients rather than simply removing them. Extensive experiments demonstrate the effectiveness of BROFL in defending against Byzantine attacks, consistently outperforming state-of-the-art baselines across various settings.
### Prerequisites
1. Python 3.9.x+
2. `pip`

### Set up
Install Python 3.9 and `pip`. We recommend using the package pyenv, which is described in [this](https://realpython.com/intro-to-pyenv/) article.
Create and enter a new virtual environment and run:
```
pip3 install -r requirements.txt
```
This will install the necessary dependencies.

### Algorithms

The datasets available for benchmarking are [MNIST](http://yann.lecun.com/exdb/mnist/), [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), and [SVHN](http://ufldl.stanford.edu/housenumbers/).

Change to `--dataset fashion` to use FashionMNIST. 
Chage to `--dataset svhn` to use SVHN.

```
python main.py --algorithm fedcvaeens --dataset mnist --num_users 50 --alpha 0.001 --local_epochs 15 --should_log 1 --z_dim 10 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 10 --uniform_range "(-1.0, 1.0)" --use_adam 1       
```

