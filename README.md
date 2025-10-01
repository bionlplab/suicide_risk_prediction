Suicide Risk Prediction with SDoH and BERT Models
====================================================

This repository implements a system for predicting suicide risk using text data enriched with Social Determinants of Health (SDoH). The pipeline supports multiple BERT variants and ensemble predictions.

Data
----
- sdoh_train.csv – training data at post level with extracted SDoH features
- sdoh_evaluate_on_leaderboard.csv – test set data for leaderboard evaluation
- data_with_instance_and_fold_labels.csv – 5-fold split data for cross-validation

Notebooks
---------
- F2_5.ipynb – 80/20 split training with RoBERTa
- F2_5_1.ipynb – 80/20 split training with BioMedBERT
- 5-fold notebooks – training with fold-based splits
- Ensemble.ipynb – ensemble voting from multiple model predictions
- final.ipynb – produces leaderboard-ready submission files

Scripts
-------
- Bert Embedding.py – generate embeddings for train/test sets (RoBERTa, BioBERT, BioMedBERT)
- predict.py – run inference on the test set using saved checkpoints

Workflow
--------
1. Generate embeddings
   Run Bert Embedding.py to create .npy files (six in total: 3 train, 3 test).

2. Train models
   - 80/20 split models: F2_5.ipynb (RoBERTa), F2_5_1.ipynb (BioMedBERT)
   - 5-fold models: run the fold-based notebooks
   - Checkpoints are saved as .pt files (available at Box link: https://wcm.box.com/s/qrqehsike93mmwdnyxge30w02sczh6pf)

3. Run predictions
   - Use predict.py with checkpoints to generate .npy outputs
   - For 5-fold models, prediction is built into the notebooks

4. Ensemble voting
   - Collect .npy outputs from all models (up to seven voters)
   - Run Ensemble.ipynb for majority-vote predictions

5. Final submission
   - Run final.ipynb with:
     - evaluate_on_leaderboard.json
     - .npy file of predicted labels
   - Produces leaderboard submission file

Requirements
------------
- Python 3.11.13
- Install dependencies via:

    pip install -r requirements.txt

Notes
-----
- Generating SDoH features is essential. See official repo: https://github.com/bionlplab/suicide_sdoh
- Ensure file paths and names are consistent with your local setup.
- Checkpoints are not stored in this repo due to size. Use the shared Box link above.
