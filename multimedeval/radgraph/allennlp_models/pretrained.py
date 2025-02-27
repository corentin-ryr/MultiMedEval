import os
import glob
from typing import Dict

from multimedeval.radgraph.allennlp.common import Params
from multimedeval.radgraph.allennlp.predictors import Predictor

from multimedeval.radgraph.allennlp_models.common.model_card import ModelCard

# These imports are included so that the model cards can be filled with default information
# obtained from the registered model classes.
from multimedeval.radgraph.allennlp_models.classification.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.coref.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.generation.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.lm.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.mc.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.pair_classification.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.rc.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.structured_prediction.models import *  # noqa: F401, F403
from multimedeval.radgraph.allennlp_models.tagging.models import *  # noqa: F401, F403


def get_pretrained_models() -> Dict[str, ModelCard]:
    """
    Returns a mapping of [`ModelCard`](/models/common/model_card#modelcard)s for all
    available pretrained models.
    """

    pretrained_models = {}
    model_card_paths = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "modelcards", "*.json"
    )

    for model_card_path in glob.glob(model_card_paths):
        model_card = ModelCard.from_params(params=Params.from_file(model_card_path))
        pretrained_models[model_card.id] = model_card
    return pretrained_models


def load_predictor(
    model_id: str, pretrained_models: Dict[str, ModelCard] = None
) -> Predictor:
    """
    Returns the `Predictor` corresponding to the given `model_id`.

    The `model_id` should be key present in the mapping returned by
    [`get_pretrained_models`](#get_pretrained_models).
    """
    pretrained_models = pretrained_models or get_pretrained_models()
    model_card = pretrained_models[model_id]
    if model_card.archive_file is None:
        raise ValueError(f"archive_file is required in the {model_card}")
    return Predictor.from_path(
        model_card.archive_file, predictor_name=model_card.registered_predictor_name
    )
