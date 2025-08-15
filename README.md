1) Create ~/.kaggle/kaggle.json with your Kaggle API key (or export KAGGLE_USERNAME & KAGGLE_KEY).
2) Install requirements: pip install -r requirements.txt
3) Train initial model (downloads dataset automatically):
   python neurons/train.py
   -> this saves models/fruit_model.pth
4) Run miner (it will load the trained model and serve axon):
   python neurons/miner.py
5) Prepare validator test images + datasets/test/labels.csv
6) Run validator:
   python neurons/validator.py
