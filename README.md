# *Unofficial* Implementation of "Latent Causal Dynamics Model for Model-Based Reinforcement Learning" (NeurIPS 2024)

This repository contains **my** version and interpretation of the paper [Latent Causal Dynamics Model for Model-Based Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-981-99-8082-6_17). I declare no affiliation with the authors of the paper, nor do I guarantee the correctness of this implementation.


## TL;DR
Learns a latent dynamics model constrained by a **causal graph G (DAG)**. The goal is to prevent spurious correlations among latent transitions that can arise in dense, fully-connected latent dynamics models.

- Causality is incorporated by learning a latent causal dynamics model among latent representations. This is formalized as a transition causal graph, which is a directed acyclic graph where nodes are latent variables at consecutive timesteps $z_t$ and $z_{t+1}$ and actions $a_t$.
- The causal structure between latent representations of consecutive timesteps is learned using a causal discovery method (typically for time series) - the PC with Momentary Conditional Independence test (PCMCI). The overall process is iterative:
  1. learn latent representations from observed states
  2. discover the causal graph among these latent variables
  3. use this structured model to aid policy learning
  4. repeat ...
- pseudocode available in paper
- no source code available, my own attempt of implementing the paper at [GitHub Repo](https://github.com/MarkusStefan/latent-causal-dynamics/tree/dev)

```
encode current observation  => obtain latent => predict next latent:    encoder(o) => z => neuralnet(z) => \hat{z'}
encode next observation     => obtain next latent:                      encoder(o') => z' 
apply loss func, backprop through ALL models:                           L(z', z)
```


![](https://github.com/MarkusStefan/latent-causal-dynamics/blob/main/assets/LCDM.PNG)
**Figure:** LCDM transition flow. The encoder $g_{\theta}$ creates latent embeddings of the image input pixel. The PCMCI algorithm is employed for discovering the edges indicating the flow of cause and effect in the causal graph $G$. Each element in the latent observation vector $z^{i}$ in the graph is modeled by an individual multi layer perceptron (MLP) $\{f_{\theta_i}\}_i$, whereby each neural network aims to predict the $i^{\text{th}}$ element in the latent vector of the next state $z_{t+1}^i$ through the values of the parent nodes (=precedents of $z_t^i$ in the causal graph at the previous timestep $t-1$), the action $a_t$ prescribed by the policy $\pi$, and a stochastic noise term $N_i \sim \mathcal{N}(0, I)$: 

$$z_{t+1}^i = f_{\theta_i}\left(PA(z_t^i), a_t, N_i\right)$$





## How to run the experiments
After installing the dependencies (see `requirements.txt`), you can run the benchmark notebook `notebooks/benchmark.ipynb` to compare LCDM with SAC and LOoP on a simple control task (e.g. CartPole). You can modify the environment and hyperparameters in the notebook.

Alternatively, run algorithms from the command line, e.g.:
```bash
python bench.py --alg lcdm --domain cartpole --task swingup --episodes 10 --steps_per_ep 1000 --seed 0
```



**Please refer to the original paper:**
```tex
@InProceedings{hao2024lcdm,
    author="Hao, Zhifeng
    and Zhu, Haipeng
    and Chen, Wei
    and Cai, Ruichu",
    editor="Luo, Biao
    and Cheng, Long
    and Wu, Zheng-Guang
    and Li, Hongyi
    and Li, Chaojie",
    title="Latent Causal Dynamics Model for Model-Based Reinforcement Learning",
    booktitle="Neural Information Processing",
    year="2024",
    publisher="Springer Nature Singapore",
    address="Singapore",
    pages="219--230",
    abstract="Learning an accurate dynamics model is the key task for model-based reinforcement learning (MBRL). Most existing MBRL methods learn the dynamics model over states. But in most cases, the relationships among states are complex because the states are affected by the interaction of various factors in the environment. Recently some works are proposed to learn the dynamics model on latent representations space. But the learned model is dense and may contain spurious associations between latent representations. To deal with these problems, we introduce a latent causal dynamics model over latent representations and provide a learning method for MBRL. Specifically, we first learn the latent representations from the observed state space. Second, we learn a latent causal dynamics model among latent representations by a causal discovery method. Finally, the latent causal dynamics model is used to aid policy learning. The above steps are iterative to update the unified loss function until convergence. Experimental results on four tasks show that the performance of our proposed method benefits from the causality and the learned latent representations.",
    isbn="978-981-99-8082-6"
}
```
