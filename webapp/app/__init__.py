import os
from flask import Flask
from config import Config
# from app import routes
from annoy import AnnoyIndex

app = Flask(__name__)
app.config.from_object(Config)

N_FEATURES = 2048

ROOT = os.path.dirname(os.path.abspath(__file__))
ann_index = AnnoyIndex(N_FEATURES, metric='angular')
ann_index.load(os.path.join(ROOT, 'knn_indices', 'training_features.ann'), prefault=True)
ann_index_multilabel = AnnoyIndex(N_FEATURES, metric='angular')
ann_index_multilabel.load(os.path.join(ROOT, 'knn_indices', 'training_features_mlrsnet_multilabel_scratch.ann'), prefault=True)

from app import routes