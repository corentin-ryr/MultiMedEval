from typing import List

from multimedeval.overrides_ import overrides

from multimedeval.radgraph.allennlp.common.util import JsonDict
from multimedeval.radgraph.allennlp.data import Instance
from multimedeval.radgraph.allennlp.predictors.predictor import Predictor


@Predictor.register("transformer_mc")
class TransformerMCPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp_models.mc.models.TransformerMC` model.
    """

    def predict(self, prefix: str, alternatives: List[str]) -> JsonDict:
        return self.predict_json({"prefix": prefix, "alternatives": alternatives})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            "no_qid", json_dict["prefix"], json_dict["alternatives"]
        )
