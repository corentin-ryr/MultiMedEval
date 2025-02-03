"""
A `Predictor` is
a wrapper for an AllenNLP `Model`
that makes JSON predictions using JSON inputs. If you
want to serve up a model through the web service
(or using `allennlp.commands.predict`), you'll need
a `Predictor` that wraps it.
"""

from multimedeval.radgraph.allennlp.predictors.predictor import Predictor
from multimedeval.radgraph.allennlp.predictors.sentence_tagger import (
    SentenceTaggerPredictor,
)
from multimedeval.radgraph.allennlp.predictors.text_classifier import (
    TextClassifierPredictor,
)
