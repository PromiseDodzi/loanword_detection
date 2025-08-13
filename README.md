# Loanword detection

This repository accompanies the paper **"Feature-Refined Unsupervised Model for Loanword Detection"** by Promise Dodzi Kpoglu. The repository contains the complete  framework, including datasets, source code, and evaluation scripts used to validate the proposed methodology.

---

## Repository Structure

The repository is organized into three main components:

- **`datasets/`** — Raw and preprocessed datasets used as input to the models  
- **`result_files/`** — Output `.tsv` files containing the results of loanword detection experiments  
- **`scripts/`** — Python scripts implementing data processing, models, and evaluation

---

## File Descriptions

### Datasets directory 

| File | Description |
|------|-------------|
| `data.tsv` | Raw dataset containing words and their transcriptions from multiple languages |
| `cleaned_data.tsv` | Preprocessed dataset after cleaning |

### Result_files directory

| File | Description |
|------|-------------|
| `sampled_data.tsv` | Data obtained when `data_samples.py` is run to obtain proportionate samples|
| `data_with_predictions_basic.tsv` | Output from `autonomous_detector.py`. Includes original data plus loanword probabilities and binary classifications |
| `data_with_predictions_scaled.tsv` | Output from `scaled_model.py`. Includes all columns from `data_with_predictions_basic.tsv` plus comparability scores |
| `data_with_predictions_uns.tsv` | Output from `scaled_model.py` on unscaled data, including loanword probabilities and classifications |


### Scripts

| File | Description |
|------|-------------|
| `clean_data.py` | Cleans and preprocesses raw data |
| `data_stats.py` | Generates summary statistics for the dataset |
| `data_samples.py` | Produces proportionate samples to test data size vs. performance |
| `basic_model.py` | Implementation of the basic autonomous loanword detection model |
| `autonomous_detector.py` | Runs the basic model on `cleaned_data.tsv` and outputs `data_with_predictions_basic.tsv` |
| `ablated_models.py` | Implements ablated versions of the autonomous model |
| `comparability_score.py` | Computes the comparability score for cross-linguistic inference |
| `scaled_model.py` | Implementation of the scaled loanword detection model |
| `reimplemented_UNS.py` | Reimplementation of **Prakhya & P (2020)** baseline model with integrated evaluation |
| `evaluator.py` | Computes performance metrics for both basic and scaled models |

---

## Reproducing experiments

### 1. Prerequisites

  **Clone the repository:**
   
    `git clone [repository-url]`
    
    `cd loanword_detection`

 **Install dependencies:**
   
    ` pip install -r requirements.txt` 


### Execution
**Data processing**

`python clean_data.py`        # Clean raw data

`python data_stats.py`   # Generate dataset statistics 

`python data_samples.py`   # Generate proportionate samples

**Baseline Model:**

`python reimplemented_UNS.py`        # Execute baseline model

**Basic  & Scaled Loanword Detection:**

`python autonomous_detector.py`  # Run the basic autonomous loanword detection model

`python scaled_model.py`            #Run the scaled loanword detection model

**Evaluation:**

`python evaluator.py`  # Obtain evaluations on the performance of the basic and scaled models

---
## Acknowledgments
This research is funded by the European Research Council (ERC) under the European Union’s Horizon Europe Framework Programme (HORIZON), grant number 101045195.

Special thanks to all members of the BANG project for their contributions, especially Aurore Montébran, who sadly passed away before the model was completed, for her constant encouragement.
