import re
from typing import List

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


class SWDEPPL(ConfigurableTask):
    VERSION = 0
    DATASET_PATH = "hazyresearch/based-swde-v2"
    DATASET_NAME = "default"

    def __init__(self, **kwargs):
        super().__init__(config={
            "metadata": {"version": self.VERSION},
            "output_type": "loglikelihood",
            "metric_list": [
                {
                    "metric": "perplexity",
                    "aggregation": "perplexity",
                    "higher_is_better": False,
                },
            ]
        })

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " " + doc["value"]
