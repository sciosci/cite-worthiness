import json
import logging
from typing import Dict, Iterable

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register('pubmed_sent_reader')
class CiteSentIdentDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
            # ,"token_characters":TokenCharactersIndexer()
            }

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                sent_json = json.loads(line)
                cur_sent = sent_json.get("cur_sent")
                label = sent_json.get("cur_has_citation")
                cur_scaled_sent_len = sent_json.get("cur_scaled_len_features", {})
                if (cur_sent is None) or (label is None):
                    print(line)
                    continue
                yield self.text_to_instance(cur_sent, cur_scaled_sent_len, label)

    @overrides
    def text_to_instance(self, cur_sent: str, cur_scaled_sent_len, label: str = None) -> Instance:
        fields = {'label': LabelField(label, skip_indexing=True)}

        cur_sent_tokens = self._tokenizer.tokenize(cur_sent)
        fields["cur_sent"] = TextField(cur_sent_tokens, self._token_indexers)

        sent_len_vec = cur_scaled_sent_len.get("values", [0.0, 0.0])
        fields["additional_features"] = ArrayField(np.array(sent_len_vec))

        return Instance(fields)
