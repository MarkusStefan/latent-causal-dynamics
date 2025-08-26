Here is the full text, figures, and plots from the provided manuscript.

### **Latent Causal Dynamics Model for Model-Based Reinforcement Learning**

**Zhifeng Hao¹,², Haipeng Zhu¹, Wei Chen¹(✉), and Ruichu Cai¹(✉)**

¹ School of Computer Science, Guangdong University of Technology, Guangzhou, China
zfhao@gdut.edu.cn, {chenweiDelight,cairuichu}@gmail.com

² College of Engineering, Shantou University, Shantou, China

**Abstract.** Learning an accurate dynamics model is the key task for model-based reinforcement learning (MBRL). Most existing MBRL methods learn the dynamics model over states. But in most cases, the relationships among states are complex because the states are affected by the interaction of various factors in the environment. Recently some works are proposed to learn the dynamics model on latent representations space. But the learned model is dense and may contain spurious associations between latent representations. To deal with these problems, we introduce a latent causal dynamics model over latent representations and provide a learning method for MBRL. Specifically, we first learn the latent representations from the observed state space. Second, we learn a latent causal dynamics model among latent representations by a causal discovery method. Finally, the latent causal dynamics model is used to aid policy learning. The above steps are iterative to update the unified loss function until convergence. Experimental results on four tasks show that the performance of our proposed method benefits from the causality and the learned latent representations.

**Keywords:** Reinforcement learning • Causal discovery • Latent representation • Dynamics model

### **1 Introduction**

Reinforcement learning has achieved impressive success in many challenging real-world domains, including video games, robot control and autonomous driving. One of the typical methods for reinforcement learning is Model-Based Reinforcement Learning (MBRL), which aims to learn an environment dynamics model through the interaction between the agent and the environment. MBRL enables an agent to predict the outcomes of its actions and learn from an imagined future. Therefore, the performance of the MBRL method is sensitive to the learned dynamics models.

Many approaches are proposed to learn an accurate dynamics model. They directly learn the dynamics model in the original observed state space. But recently, some works have argued that modeling all observed state variables is expensive and unnecessary, such as PlaNet and ALM. Because in most real-world situations, the state variables are very high-dimensional, which leads to hard training. In addition, some observed state variables are not related to the task, without semantic information. Thus, they focus on modeling the task-related information in the environment from states, which contain semantic information. Those are referred to as latent representations.

The dynamics model learned by existing methods tends to be dense, where all latent representations are interconnected. This implies that when predicting the value of each latent representation at a given timestamp, the model relies on the current action and all latent representations from the previous timestamp. However, not all latent representations are dependent on each other. For example, in Fig. 1 (a), the car’s color is independent of the car’s speed. In this case, the correlation between the color and the speed is spurious in the dynamics model. If using this model to predict the car’s speed by taking the car’s color at the previous time into consideration, it may lead to incorrect predictions.

To learn a good dynamics model, two main challenges need to be tackled: 1) learning task-relevant latent representations from states and 2) removing spurious relationships between latent representations to recover the true dynamics model. Based on the above analysis, we find that the environmental information received by an agent can be categorized into two kinds: 1) task-relevant information; 2) task-irrelevant or weakly relevant information. Therefore, a good latent representation should only capture task-relevant information. As for the second challenge, spurious relationships between latent representations arise when they are not truly dependent. These spurious relationships can be detected through causal discovery methods. Thus, to overcome these challenges, we propose a Latent Causal dynamics model learning method. First, we obtain task-relevant latent representations by constraining them to be associated with state changes and received rewards. Second, to remove the spurious correlations between the latent representations, we use causal discovery methods for recovering the causal structure between the latent representations, which is used to obtain the correct dynamics model as well as the reward model. These steps are iteratively performed to update the latent representation learning, latent causal dynamics model learning, and policy learning until convergence is achieved.

### **2 Preliminary**

**2.1 Reinforcement Learning**
In this paper, we consider infinite-horizon Markov Decision Processes (MDP), which is described by a tuple (S, A, p, R, γ, ρ₀). Here, S ∈ Rⁿ and A ∈ Rᵐ represent continuous state and action spaces respectively, p(sₜ₊₁|sₜ, aₜ) denotes the transition distribution, R : S × A → R or S → R is a reward function, γ is the discount factor which lies in (0, 1), and ρ₀ represents the initial state distribution. Model-based reinforcement learning is one of the typical methods of reinforcement learning. It aims to construct a model of the transition distribution, denoted as pθ(sₜ₊₁|sₜ, aₜ). This model is learned using data collected from interacting with the environment. Additionally, a reward model is learned through supervised learning to approximate the true reward function R. In MBRL, the dynamics and reward functions are usually assumed to be unknown.

**2.2 Structural Causal Model**
The Structural Causal Model (SCM) represents causal relationships between two variables as their functional dependencies. The definition is as follows:

**Definition 1.** (Structural Causal Model). A Structural Causal Model (SCM) is a tuple (V, G, F, N), where
- V is a set of random variables involved in the model;
- G is a directed acyclic graph (DAG), in which the node corresponds to variables in V and the edges indicate the causal connections between variables.
- F is a set of structural functions that describe the functional relationships between the variables;
- N is a set of noise terms associated with each variable.

Furthermore, for each variable vᵢ ∈ V, vᵢ is associated with a function that describes how it is influenced by its directed causes, which is formulated as: vᵢ = fᵢ((vₚₐᵢ), nᵢ), where vᵢ ∈ V, fᵢ ∈ F, nᵢ ∈ N, and vₚₐᵢ represents parents of vᵢ.

### **3 Latent Causal Dynamics Model Learning**

In this section, we introduce a method for learning a latent causal dynamics model (LCDM), and then derive a policy through planning based on the model.

**3.1 Latent Causal Dynamics Learning**
Given a fully observed Markov Process with V = {sₜᴺ, aₜ, sₜ₊₁ᴺ, rₜ₊₁}, where sₜᴺ is denoted as the state variables sᴺ at timestamp t and N represents the dimensional of states. We aim to learn a good dynamics model for MBRL. In MBRL, if an agent observes all the information in the environment, then we can directly extract all the task-relevant information from the information observed by the agent. In the real world, however, we can not observe all information. Hence, our focus is on recovering or extracting the task-relevant information from what we observe. This kind of information is implied in the observed state variables, which may represent a combination of multiple state variables. Take Fig. 1 as an example, the speed of the car is a task-relevant representation that is a combination of multiple sensor data.

Specifically, the extracted information is referred to latent representations, which contains the task-related latent representations defined as Z and task-independent latent representations defined as U. Then, we only need to care about Z (|Z| = M). Let zₜ denote the latent representation z that is extra from S at time step t. The generation process of representations can be represented by a causal graph G = {Z, E}, where E are causal edges that describe the causal relationships between zᵢᵗ and zᵢᵗ⁺¹, i ∈ {1, 2, ..., M}.

Therefore, the aim of latent causal dynamics learning includes 1) learning the latent representations and 2) discovering the causal structure among the latent representations. Regarding the first task, we can extract the representations from the observed states. Inspired by, we use an encoder gθ to model the state-to-latent representation mapping, then use a set of neural network to model the latent representations transition.

To ensure the stability of the learning process of latent representations, we use the L2 distance between zₜ₊₁ and the gθ(sₜ₊₁) as a consistency loss. This loss encourages the latent representations to remain consistent and prevents them from diverging during the learning process. The reason why we do not use a decoder to reconstruct the state here is that we do not require the latent representations to encapsulate all the information present in the environment. Thus, the learning function is formalized as follows:

Lθ = Σ ||fθ(zₜ, aₜ, G), gθ(sₜ₊₁)||₂     (1)

where G is a causal graph that contains the relationships among Z.

Recently, some studies consider that the representations are correlated with each other. But actually, these correlations may be spurious and not truly indicative of causal relationships. For example, in Fig. 1, the speed and color of the car are actually independent of each other. This kind of independence can be detected by causality since causality describes the mechanism of data generation. Hence, we incorporate causal structure as a constraint in the latent representation learning procedure. This ensures that the learned latent representation transition function is more reasonable. Additionally, our reward model is built upon latent representations, which enables the latent representations to capture and incorporate as much task-relevant information as possible.

Learning a causal structure among representations involves determining whether a causal edge exists between two latent representations. That is, determining whether zᵢᵗ → zⱼᵗ⁺¹ or zⱼᵗ → zᵢᵗ⁺¹ is corresponding to the truth causal relationship. To make this inference, certain assumptions are required.

- **Assumptions A1.** Causal Markov and Faithfulness assumptions in the underlying dynamics.
- **Assumptions A2.** The latent dynamics is Markovian and stationarity.
- **Assumptions A3.** The edge zᵢᵗ → zⱼᵗ⁺¹ exists for all variables z mechanism for all transitions
- **Assumptions A4.** No simultaneous or backward edges in time, i.e., for all i, j, zᵢᵗ ↛ zⱼᵗ and zᵢᵗ⁺¹ ↛ zⱼᵗ.

Based on the above assumptions, we can identify the causal relationship between representations, which is guaranteed by the following theorem.

**Theorem 1.** Suppose Assumptions A1-A4 hold. Let {aₜ, zₜ\zᵢᵗ} = {aₜ, z₁ᵗ, ..., zᵢ₋₁ᵗ, zᵢ₊₁ᵗ, ..., zₘᵗ}. For any two latent variables zᵢᵗ and zⱼᵗ, if zᵢᵗ ⊥̸⊥ zⱼᵗ⁺¹|{aₜ, zₜ\zᵢᵗ}, then zᵢᵗ → zⱼᵗ⁺¹.

*Proof.* We need to show that if zᵢᵗ ⊥̸⊥ zⱼᵗ⁺¹, then the condition zᵢᵗ ⊥ zⱼᵗ⁺¹|{aₜ, zₜ\zᵢᵗ} or the assumption is violated. Assume there is no direct edge from zᵢᵗ to zⱼᵗ⁺¹ but they are dependent, i.e., zᵢᵗ ↔ zⱼᵗ⁺¹ and zᵢᵗ ⊥̸⊥ zⱼᵗ⁺¹, then there is a latent confounder variable Qᵗ' with the confounding path zᵢᵗ ←-- Qᵗ' --→ zⱼᵗ⁺¹ where t' < t as we assume there is no simultaneous edge and backward edge. If Qᵗ' exists in this case, then it violates the condition because we condition on {aₜ, zₜ\zᵢᵗ} which block all possible paths for Qᵗ' --→ zⱼᵗ⁺¹ and thus d-separate zᵢᵗ and zⱼᵗ⁺¹. Therefore we show that if zᵢᵗ ⊥̸⊥ zⱼᵗ⁺¹|{aₜ, zₜ\zᵢᵗ}, then zᵢᵗ → zⱼᵗ⁺¹.

This theorem demonstrates that the causal relationship between two variables can be inferred by the Conditional Independence Test (CIT) method. Several CIT methods can be employed, including conditional mutual information and Kernel-based Conditional Independence Test (KCIT). In this paper, we use PCMCI (PC with Momentary Conditional Independence test) to recover the causal graph. The consistency of the learned causal graph and the true causal graph is guaranteed by the following theorem, which has been proved in.

**Theorem 2.** (Consistency) Let X be a Markov decision process with true transition causal graph G as defined in Definition 2 and Ĝ be the estimated graph by PCMCI method with a consistent conditional independence test. Suppose Assumptions A1-A4 hold, Ĝ = G.

Previous research has shown that the learned causal graph among representations can be connected to the transition in reinforcement learning. Inspired by, we define the transition causal graph as follows.

**Definition 2.** (Transition Causal Graph) We define a Transition Causal Graph as a directed graph denoted by G, where the vertices are divided into two disjoint sets: U = {Aₜ, Zₜ} and V = {Zₜ₊₁}. Let Aₜ represent action nodes at time step t, Zₜ represent latent representation variables at step t, and Zₜ₊₁ denote the latent representation variable at step t+1. All edges start from set U and end in set V.

The Transition Causal Graph captures the causal relationships between latent representation variables at two consecutive time steps. It signifies that the values of a latent representation variable at time step t + 1 depend on the values of the same latent representation variable at time step t. This temporal dependency in the graph reflects the dynamics of the system and how the latent representations evolve over time. By modeling these causal relationships, we can better understand and predict the changes in latent variables from one time step to the next, which is essential for planning, decision-making, and learning in reinforcement learning scenarios. Combined with the Definition 1, the latent representation transition, p(zₜ₊₁|zₜ, aₜ), can be expressed as follows:

p(zₜ₊₁|zₜ, aₜ) = Π p(zᵢᵗ⁺¹|PA(zᵢᵗ⁺¹), aₜ))     (2)

where PA(zᵢᵗ⁺¹) is a set of parent of node zᵢᵗ⁺¹ in the transition causal graph G. From this equation, we can find that p(zₜ₊₁|zₜ, aₜ) essentially approximates a collection of functions fᵢ following the transition Causal Graph G, which take as input the values of PA(zᵢᵗ⁺¹) and output the value of zᵢ. In practice, we use a collection of neural networks {fθᵢ}ᵢ₌₁ᴺ to model the latent transition corresponding to transition Causal Graph G, which is formalized as:

zᵢᵗ⁺¹ = fᵢ(PA(zᵢᵗ), aₜ, Nᵢ),     (3)

where PA(zᵢᵗ) represents the values of all parents of node zᵢᵗ in transition Causal Graph G, and Nᵢ is noise terms that are assumed to follow a Gaussian distribution, e.g. Nᵢ ∼ N(0, I).
Based on the above analysis, the loss function for learning the latent dynamics model is as follows:

L(θ, φ) = Σ [||fθ(zₜ, aₜ, G), gφ(sₜ₊₁)||₂ + ||rφ(zₜ, aₜ), rₜ||₂]     (4)

where ||.||₂ represents Mean-Square Error, fθ denotes the transition model of latent representation zₜ with parameters θ, rφ denotes the predictive model of rₜ with parameters φ. Notice that only the initial time, e.g. s₀ has a gradient during the encoder stage, whereas other time states have no gradient during the decoder stage.

Thus, the model can be trained by the following procedure. The procedure starts with a completely directed graph, and takes the state at the initial moment to obtain the initial latent representation through the encoder gθ. Then, it utilizes the causal graph as a constraint to learn the dynamics model between the latent representations. Additionally, it learns a reward model based on latent representations. The agent takes action to interact with the environment and generates data to update our causal graph. This subsequent process is repeated until convergence. This training procedure is characterized in Fig. 2.

**3.2 Policy Learning**
The learned causal dynamics model plays a crucial role in improving policy learning. Based on the learned model, we employ the model predictive control (MPC) as the planning algorithm. MPC is an iterative and model-based control approach. After taking an action, MPC first generates a set of candidate action sequences. Then, it evaluates how good the result of each candidate sequence can be based on the current state. Finally, it picks the first action of the action sequence with the best result to execute. By iteratively repeating this process, the agent can make informed decisions at each time step, optimizing its actions based on the learned model’s predictions of the future states and rewards.

Specifically, we use Model Predictive Path Integral (MPPI) control algorithm. MPPI is a variation of the MPC algorithm that iteratively adjusts parameters for a range of distributions. It does so by employing an importance-weighted average of the expected returns of the top-k sampled trajectories. Because MPPI allows for changes in the drift and diffusion terms of stochastic diffusion processes, it can help mitigate the rollout error even when our learned latent causal dynamics model has a slight distribution shift with respect to the ground truth transition distribution.

### **Algorithm 1. LCDM Training**

**Require:** Latent Transition model fθ, Encoder gθ
1: Initialize latent causal graph G as a complete directed graph
2: **while** θ not converged **do**
3:   // Policy learning from planning
4:   **for** step t=0...T **do**
5:     Select action given by aₜ = Planner(fθ, gθ(sₜ)).
6:     Execute aₜ in the environment and observe reward rₜ and new state sₜ₊₁.
7:     Store the transition(sₜ, aₜ, rₜ, sₜ₊₁) in Trajectory Buffer Bₜ.
8:   **end for**
9:   // Latent transition model learning
10:  Update fθ(G) and gθ via Eq. (4) with Bₜ
11:  // Estimate Latent Causal graph
12:  latent causal graph G ← PCMCI(Bₜ).
13: **end while**

Therefore, combining the latent causal dynamics model learning and policy learning, our proposed method is summarized in Algorithm 1.

### **4 Experiments**

In this section, we conducted experiments to evaluate our proposed method on some diverse and challenging continuous tasks from different environments that are described in Sect. 4.1. We aim to answer these questions:

- **Q1.** How does the performance of our method compare to that of state-of-the-art model-based and model-free methods in different environments?
- **Q2.** How useful are the learned representations for our dynamics model?
- **Q3.** How helpful is learning the causal structure between representations for our method?

**Evaluation Metric.** We use the average cumulative reward across 5 seeds with random initialization to measure the performance of methods.

**4.1 Environments and Baselines**
**Environment Details.** We use environments (given in Fig. 3) from Deep-Mind (DM) Control Suite that is a standard benchmark for continuous control. In these experiments, we set the episode length as 1000 steps and repeat an action twice for all tasks. Every experiment is run with different random seeds.

**Baselines.** We evaluate our method against the following methods as baselines:
- **SAC:** Soft Actor-Critic is a model-free algorithm derived from maximum entropy RL. We choose SAC as our main point of comparison due to its popularity and strong performance on DMControl.
- **LOOP:** A hybrid algorithm that combines H-step lookahead policies with a learned model and a learned terminal value function using a model-free off-policy algorithm.
- **LCDM w/o representation:** This variant removes the representation learning procedure by setting the encoder gθ as a identify function.
- **LCDM w/o causal structure:** This variant removes the causal graph between the learned latent representations.

**4.2 Q1: Performance Comparison**
Figure 4 shows the reward of our method and baseline methods in five tasks. From the results, we observe that: 1) In the expected Quadruped-run task, all methods converge quickly. Our method achieves higher cumulative rewards than all baseline methods in all the environments, with faster convergence. 2) Our method performs better in both low-dimensional and high-dimensional state space environments. Since SAC is a model-free algorithm, it needs to interact with the environment many times and explore the environment fully before the performance gets better and better. From Fig. 4, we can see that SAC can learn a better policy with fewer interactions for simple environments. But for complex environments such as Quadruped-run, SAC needs a lot of interactions to fully explore the environment before it obtains a better policy.

Our method can achieve better performance because it can filter out some irrelevant state information by the learned representations and obtain a powerful World Model by quickly learning the causal structure between representations, which helps to obtain the best reward.

**4.3 Q2: Will the Learned Representations Be Useful for Our Dynamics Model?**
We design an ablation version of our method, LCDM w/o representation, which uses the state space instead of the latent representations to build the world model and the reward predictor. In these experiments, the shaded areas are set as 95% confidence intervals. For each task, we run 5 trials and use the average reward of these trials as the result. The results are shown in Fig. 5. Compared with LCDM w/o representation, LCDM method earns a higher accumulative reward than its ablation version in all environments. In fact, although the LCDM w/o representation gets access to all observation state information, it may be confused or sensitive since part of the observable state dimensions may not be helpful for predicting rewards or performing tasks. Instead, the learned representations help to build a world model by ignoring some states that have no or little impact on the task and aggregating those that have a great impact on the task.

**4.4 Q3: Is the Learned Causal Structure Between Representations Helpful for Our Method?**
To analyze the impact of learning causal structure between representations, we construct an ablation version of our method, LCDM w/o causal structure, which used a fully connected structure to replace causal structure. According to the results shown in Fig. 5, our proposed method obtains a higher cumulative reward than LCDM w/o causal structure method. In fact, although we could not explain what representations we learned, the result reflects that there was a causal relationship between the representations. When we did not consider the causal relationship between them, the world model learned directly from the representations was incorrect, which led to a reduction in the generalization ability of the model. Thus, the model with causal structure among presentations helps to achieve better rewards while obtaining good generalizability.

### **5 Conclusion**

In this paper, we present a novel model-based reinforcement learning method that learns a latent causal dynamics model from observations. Our method leverages both causal structure and representation learning to capture the essential dependencies information and mechanisms of the underlying system. We have shown that our method can learn accurate and robust world models that can generalize well to unseen states and be resilient to subtle changes. We have also demonstrated the effectiveness of our method on several challenging continuous control tasks, where it outperforms state-of-the-art model-based and model-free baselines. Our work opens up new possibilities for incorporating causal reasoning and representation learning in reinforcement learning.

---
### **Figures and Plots**

**Figure 1: Two latent dynamics models on latent representations**
This figure illustrates the core concept of the paper.
*   **(a)** Shows the data generating process where observed state variables (s¹...sᴺ) are influenced by latent causal variables (z¹...zᴹ).
*   **(b)** Depicts a standard latent dynamics model where all latent representations at time `t` (e.g., Car color, Car Speed) are densely connected to predict all latent representations at `t+1`. This can create spurious correlations.
*   **(c)** Shows the proposed latent dynamics model. It uses a causal structure to model the transitions. For instance, it correctly identifies that only a few variables at time `t` (like Car Speed and the Agent's action) actually cause the Car Speed at `t+1`, while the Car color is independent. This removes spurious connections.

![Figure 1](https://storage.googleapis.com/ask-gemini-docs/images/figure1_2024-05-15_16-09-41.png)

**Figure 2: Latent Causal Dynamics Models (LCDM) learning procedure**
This figure shows the workflow of the model training.
1.  An initial observation `s₀` is fed into an `Encoder gθ` to get the initial latent representation `z₀`.
2.  A policy decides on an action `a₀`.
3.  The causal structure `CG` (Causal Graph) is used as a constraint to predict the next latent state `z₁`.
4.  This process repeats for subsequent steps (`z₁, a₁, z₂`, etc.).
5.  An L2 norm consistency loss is applied between the predicted latent state and the encoded latent state from the actual next observation (e.g., `||z₁' - z₁||₂`).

![Figure 2](https://storage.googleapis.com/ask-gemini-docs/images/figure2_2024-05-15_16-09-54.png)

**Figure 3: An illustration of DM Control Suite environments**
This figure displays images of the five environments from the DeepMind Control Suite that were used for the experiments. The tasks are selected to cover a range of complexities in state and action spaces. The environments shown are: Fish, Walker, Quadruped, Reacher, and Hopper.

![Figure 3](https://storage.googleapis.com/ask-gemini-docs/images/figure3_2024-05-15_16-10-06.png)

**Figure 4: DMControl tasks. Average returns of our method and baselines on 6 tasks from DMControl.**
This figure contains six plots showing the performance (average reward over steps) of the proposed method (LCDM, in green) against baselines (sac, loop) on various tasks.
*   **Tasks:** `quadruped-run`, `fish-swim`, `walker-run`, `hopper-hop`, `walker-walk`, `reacher-hard`.
*   **Observation:** In all tasks, the LCDM method (green line) generally achieves a higher cumulative reward and converges faster than the baseline methods. The shaded areas represent 95% confidence intervals.

![Figure 4](https://storage.googleapis.com/ask-gemini-docs/images/figure4_2024-05-15_16-10-18.png)

**Figure 5: Ablation Result. Average reward of our method and our ablation methods on 4 state-based continuous control tasks.**
This figure presents four plots from an ablation study to test the importance of the main components of the model.
*   **Methods:**
    *   `LCDM` (blue): The full proposed method.
    *   `LCDM w/o representation` (orange): Ablation without learning latent representations.
    *   `LCDM w/o causal structure` (green): Ablation without using the causal graph (i.e., a fully connected graph).
*   **Tasks:** `quadruped-run`, `fish-swim`, `walker-run`, `hopper-hop`.
*   **Observation:** The full LCDM model (blue line) consistently outperforms both ablation versions, demonstrating that both the learned latent representations and the learned causal structure are crucial for achieving the best performance.

![Figure 5](https://storage.googleapis.com/ask-gemini-docs/images/figure5_2024-05-15_16-10-31.png)