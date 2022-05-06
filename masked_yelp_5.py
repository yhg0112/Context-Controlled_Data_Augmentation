# coding=utf-8
# Copyright 2022 Hyeongu Yun

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Masked yelp-5 dataset for data augmentation."""
import os
import json

import datasets
from datasets.tasks import TextClassification


_DESCRIPTION = """\
Masked yelp-5 dataset for data augmentation.\
"""

_DATA_FILES = {
    "masked_yelp_4": {"train_file": "./data/text_gen/masked_yelp_4_train.json",
                      "test_file": "./data/text_gen/masked_yelp_4_valid.json"},
    "masked_yelp_5": {"train_file": "./data/text_gen/masked_yelp_5_train.json",
                      "test_file": "./data/text_gen/masked_yelp_5_valid.json"},
    "masked_yelp_6": {"train_file": "./data/text_gen/masked_yelp_6_train.json",
                      "test_file": "./data/text_gen/masked_yelp_6_valid.json"},
    "masked_yelp_7": {"train_file": "./data/text_gen/masked_yelp_train_7.json",
                      "test_file": "./data/text_gen/masked_yelp_7_valid.json"},
}

class MaskedYelpReviewsConfig(datasets.BuilderConfig):
    """BuilderConfig for MaskedYelpReviews."""

    def __init__(self, **kwargs):
        super(MaskedYelpReviewsConfig, self).__init__(**kwargs)


class MaskedYelp(datasets.GeneratorBasedBuilder):
    """Masked Yelp-5 reviews dataset."""

    VERSION = datasets.Version("0.1.1")
    BUILDER_CONFIGS = [
        MaskedYelpReviewsConfig(
            name="default",
            version=VERSION,
            description="Masked Yelp 5 reviews, with 0.4 masking probabilty.",
        ),
        MaskedYelpReviewsConfig(
            name="masked_yelp_4",
            version=VERSION,
            description="Masked Yelp 5 reviews, with 0.4 masking probabilty.",
        ),
        MaskedYelpReviewsConfig(
            name="masked_yelp_5",
            version=VERSION,
            description="Masked Yelp 5 reviews, with 0.5 masking probabilty.",
        ),
        MaskedYelpReviewsConfig(
            name="masked_yelp_6",
            version=VERSION,
            description="Masked Yelp 5 reviews, with 0.6 masking probabilty.",
        ),
        MaskedYelpReviewsConfig(
            name="masked_yelp_7",
            version=VERSION,
            description="Masked Yelp 5 reviews, with 0.7 masking probabilty.",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "label": datasets.Value("int8"),
                "text": datasets.Value("string"),
                "masked_text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        filenames = _DATA_FILES[self.config.name] if self.config.name else _DATA_FILES["masked_yelp_7"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filenames['train_file'],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": filenames['test_file'],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, 'r') as f:
            data = json.load(f)
            for id_, example in enumerate(data):
                yield id_, example

