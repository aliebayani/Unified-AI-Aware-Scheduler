# Unified AI-Aware Scheduler Prototype

A complete Phase-1 implementation and extension of:

**Sampling-Based Multi-Job Placement for Heterogeneous Deep Learning Clusters**

This project includes:
- JPS scheduler implementation  
- Automatic tuning of α and β  
- Fairness, entropy, and utilization analysis  
- Full experiment suite with 10+ research-quality plots  
- Phase-2 architecture plan for NCCL-aware and operator-fusion scheduling  

---

## Features
- Synthetic heterogeneous GPU cluster simulation (T4, P100, V100)
- Full category enumeration + matching-market worker assignment
- JPS sampling-based scheduler
- Automatic α–β tuning
- Extended evaluation:
  - JCT CDF
  - Fairness–JCT curves
  - Category variance
  - Category entropy
  - Worker utilization heatmap
  - JCT vs number of workers
  - Random vs uniform vs JPS comparison
- Reproducible research pipeline (`run_all.py`)

---

## Project Structure

Project Root
- sim.py
- auto_tune.py
- evaluation.py
- run_all.py
- plots/
  - category_entropy.png
  - category_variance.png
  - fairness_tradeoff.png
  - grid_heatmap.png
  - jct_cdf.png
  - jct_vs_workers.png
  - sampler_comparison.png
  - worker_heatmap.png
- results/
  - fairness_metrics.txt
  - summary_table.csv
- autotune_trials.png
- auto_tune_history.csv
- README.md          

---

## Requirements
- Python **3.8+**

Install all dependencies:

pip install -r requirements.txt

### `requirements.txt`
- contourpy==1.3.3
- cycler==0.12.1
- fonttools==4.61.0
- kiwisolver==1.4.9
- matplotlib==3.10.7
- numpy==2.3.5
- packaging==25.0
- pandas==2.3.3
- pandoc==2.4
- pillow==12.0.0
- plumbum==1.10.0
- ply==3.11
- pyparsing==3.2.5
- python-dateutil==2.9.0.post0
- pytz==2025.2
- pywin32==311
- seaborn==0.13.2
- six==1.17.0
- tzdata==2025.2


### **2. Run Auto-Tuning**
python auto_tune.py

### **3. Generate All Plots and Tables**
python run_all.py
Outputs appear in:
- `plots/`
- `results/`

## Output Summary

### Plots (in `plots/`)
- Grid search heatmap  
- Fairness–JCT tradeoff  
- JCT CDF  
- Category size variance  
- Category entropy  
- Worker utilization heatmap  
- JCT vs number of workers  
- Sampler comparison  
- Auto-tuning trials  

### Results (in `results/`)
- Jain fairness index + JCT spread  
- Summary table of methods  


