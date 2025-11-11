# Your Paper Title Here

<!-- 在这里放一些徽章 (Badges)，非常专业。可以从 shields.io 生成 -->
<!-- 例如: Paper, Code License, Python Version -->
<!-- <p align="center">
  <a href="[你的论文PDF链接，例如arXiv链接]">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square" alt="Paper">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" alt="Python">
</p>

Official PyTorch implementation for the paper: **"[Your Paper Title]"**.

[作者1]([作者1主页链接]), [作者2]([作者2主页链接]), ...
[你的大学或机构] -->

### **Abstract**
> Accurate path loss prediction is critical for designing future wireless networks, yet traditional models struggle to cope with the complexity of diverse urban environments. While deep learning (DL) has emerged as a powerful solution, existing models often fail to generalize across these environments and suffer from catastrophic forgetting when sequentially updated, limiting their practical deployment. To address these critical challenges, this paper introduces, for the first time, a continual learning framework specifically designed for path loss prediction. Our proposed method enhances a standard encoder-decoder architecture with a novel structure-optimized adaptation strategy. This strategy incorporates three key components: 1) a lightweight Scalable Task-Adaptive Module (STAM) to capture environment-specific features in the encoder; 2) a Residual Attention Map Module (RAMM) to refine feature representations between the encoder and decoder; and 3) a multi-decoder architecture to provide dedicated reconstruction pathways for each task. By freezing the shared backbone and only training these lightweight, task-specific modules, our framework efficiently adapts to new environments while preserving previously acquired knowledge. Extensive experiments on three distinct urban datasets demonstrate that our method not only significantly mitigates catastrophic forgetting but also exhibits positive forward transfer, where knowledge from previous tasks enhances learning on new ones. Furthermore, our framework shows remarkable robustness to the order of sequential tasks and achieves a superior balance between predictive accuracy and parameter efficiency compared to a wide range of continual learning baselines.

---

## 🌟 Visual Abstract / Model Architecture

<!-- 在这里放一张最能代表你工作的图，比如你的模型架构图。这是吸引读者的利器！ -->
<p align="center">
  <img src="figure/overall.svg" width="800">
  <br>
  <em>An overview of our proposed framework. It consists of a shared backbone, task-specific modules (STAM, RAMM), and a multi-decoder head.</em>
</p>


## ✨ Main Contributions
*   To the best of our knowledge, this is the first work to introduce continual learning to the domain of path loss prediction to address model adaptability and catastrophic forgetting.

*   We propose a continual learning framework based on a dynamic architectural approach, which integrates novel task-specific modules (STAM, RAMM) into a multi-decoder structure. This allows the model to expand its capacity for new tasks, enabling rapid adaptation while preserving shared knowledge.
*   We provide a comprehensive empirical evaluation on multiple urban datasets, establishing a new benchmark and demonstrating our method's superior performance in mitigating forgetting, enabling knowledge transfer, and balancing accuracy with efficiency.


## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ErwinsWu/CL4PL.git
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    conda create -n your_env_name python=3.8
    conda activate your_env_name
    ```

3.  **Install dependencies:**
    We provide a `requirements.txt` file for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Our project uses `.yaml` configuration files located in the `configs/` directory to manage all experiments. To run training or evaluation, you first need to configure the appropriate `.yaml` file and then execute the corresponding Python script.

### 1. Configure Your Experiment

Navigate to the `configs/` directory. You will find template configuration files for different models and scenarios. 

### 2. How to Run Experiments
#### A. Training a Model (Continual Learning)
To train a model (e.g., LRRA) on a sequence of tasks:

1.  **Configure:**
Open a config file like configs/lrra.yaml. Set the model.tasks list and other hyperparameters as needed.

2.  **Run Training:**
    ```bash
    python train.py --config 'configs/lrra.yaml'
    ```

    The script will sequentially train the model on each environment specified in the tasks list. Checkpoints and logs will be saved in the directories defined in the config.
#### B. Evaluating a Trained Model
1.  **Configure:** Use an evaluation config file or modify a training one. Ensure evaluation.model_path points to the directory with your saved checkpoints.
2.  **Run Evaluation:**
    ```Bash
    python eval.py --config 'configs/lrra.yaml'
    ```
    The evaluation results will be printed to the console.
