# BEACOF: Belief-Driven Multi-Agent Collaboration

[![Conference](https://img.shields.io/badge/WWW-2026-brightgreen)](https://www2026.thewebconf.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

**Official implementation for the paper:** > **Belief-Driven Multi-Agent Collaboration via Approximate Perfect Bayesian Equilibrium for Social Simulation** > *Weiwei Fang, Lin Li, Kaize Shi, Yu Yang, and Jianwei Zhang* > **The Web Conference (WWW) 2026**

---

## 📖 Overview

High-fidelity social simulation demands agents capable of authentically replicating the dynamic spectrum of human interaction. Current Multi-Agent Systems (MAS) often adhere to static interaction topologies, leading to "groupthink" (in pure cooperation) or deadlocks (in pure competition).

**BEACOF** (**Be**lief-Driven **A**daptive **Co**llaboration **F**ramework) is a novel framework that models social interaction as a dynamic game of incomplete information. Inspired by **Approximate Perfect Bayesian Equilibrium (PBE)**, BEACOF enables agents to:

1.  **Maintain Beliefs:** Iteratively refine probabilistic beliefs about peer capabilities based on interaction history.
2.  **Adaptive Strategy:** Autonomously switch between **Cooperation**, **Competition**, and **Coopetition** (collaborative competition).
3.  **Ensure Rationality:** Make sequentially rational decisions under uncertainty without needing full information.

![Framework Overview](assets/framework_overview.png)
*(Note: Please upload Figure 2 from the paper to an `assets` folder and reference it here)*

---

## 🚀 Key Features

* **Dynamic Strategy Switching:** Agents are not fixed to a single role; they transition dynamically between cooperative knowledge synthesis and competitive critical reasoning.
* **Gaussian Belief Updates:** Implements a tractable parametric Bayesian update mechanism with a forgetting factor to track non-stationary peer capabilities.
* **Meta-Agent Coordination:** A centralized coordinator estimates contextual payoffs and evaluates message quality to drive belief evolution.
* **Multi-Scenario Generalization:** Validated across three distinct social interaction archetypes:
    * ⚖️ **Adversarial:** Judicial/Court Debate.
    * 🏥 **Mixed:** Medical Q&A (MedQA).
    * 🗣️ **Open-Ended:** Persona-based Social Chat.

---

## 🛠️ Installation

### Prerequisites
* Python 3.9+
* [Ollama](https://ollama.com/) (for local LLM inference as described in the paper)

### Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/WUT-IDEA/BEACOF.git](https://github.com/WUT-IDEA/BEACOF.git)
    cd BEACOF
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Setup (Ollama)**
    Ensure you have the backbone models pulled via Ollama. We utilize the following models in our experiments:
    ```bash
    ollama pull llama3.1
    ollama pull gemma2:9b  # (Check specific tag for Gemma3 equivalent)
    ollama pull qwen2.5:32b # (Check specific tag for Qwen3 equivalent)
    ```

---

## 📂 Project Structure

```text
BEACOF/
├── data/
├── src/
│   ├── AgentsCourt/
│   ├── mechanism/
│   └── prompts/
│	├── interaction.py                 # Entry point for simulations
│	└── ollama.py
└── README.md
```

------

## 🏃 Usage

You can run simulations for different scenarios using `main.py`.

### Start Simulation

```
# 1. Court Debate Simulation
python main.py court

# 2. Medical Consultation Simulation
python main.py medqa

# 3. Daily Chat Simulation
python main.py persona
```

------

## 🧩 Methodology Highlights

### The Loop

1. **Payoff Generation:** The Meta-Agent generates contextual payoffs $U_t$ based on interaction history.
2. **Action Prediction:** Probability distributions over collaboration types are predicted.
3. **Strategic Action:** Participant agents calculate an approximate **Best Response** ($c_i^*$) maximizing expected utility.
4. **Belief Update:** Agents update Gaussian beliefs about peers using the evaluation $e_j^t$ and confidence $\omega$, modulated by a forgetting factor $\lambda$.

### Equation: Belief Update

$$b_{i}^{t}(j) = \frac{\omega_{i}^{t-1}(j) \cdot b_{i}^{t-1}(j) + \omega_{j}^{t} \cdot e_{j}^{t}}{\omega_{i}^{t-1}(j) + \omega_{j}^{t}}$$

------

## 📝 Citation

If you find this code or our paper useful, please cite:

```
@inproceedings{fang2026beacof,
  title={Belief-Driven Multi-Agent Collaboration via Approximate Perfect Bayesian Equilibrium for Social Simulation},
  author={Fang, Weiwei and Li, Lin and Shi, Kaize and Yang, Yu and Zhang, Jianwei},
  booktitle={Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year={2026},
  publisher={ACM},
  address={Dubai, United Arab Emirates}
}
```

## 📧 Contact

For any questions, please contact:

- **Weiwei Fang**: `311137@whut.edu.cn`
- **Lin Li**: `cathylilin@whut.edu.cn`

------

*This work is supported by the National Natural Science Foundation of China and The Education University of Hong Kong.*