# MirrorMinds-LLMDetector
Code repository of paper "Mirror Minds : An Empirical Study on Detecting LLM-Generated Text via LLMs" published in DAIGenC25@COLING 2025.
This repository is a fork of the original [MirrorMinds](https://github.com/shubhamgpt007/MirrorMinds-LLMDetector/) project, modified to provide a minimal, standalone library that can be easily integrated into other projects. 
It also introduces support for configuring different LLMs.

## 🚀 Getting Started

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone <url_to_this_repository>
cd MirrorMinds-LLMDetector
```

### 2. Install the required packages:

#### 2.1 Create a virtual environment:
```bash
python -m venv venv
```
#### 2.2 Activate the virtual environment:
    - On Windows:
```bash
venv\Scripts\activate
```
    - On macOS and Linux:
```bash
source venv/bin/activate
```
#### 2.3 Update pip:
```bash
pip install --upgrade pip
```

#### 2.4 Install the required packages:
```bash
pip install -e .
```

### 3. Run the main script:
```bash
python main.py
```

To see a step-by-step display of the results, open and run the `display_results.ipynb` notebook.

