import json
import logging
from typing import Dict, Iterable

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register('pubmed_contextual_reader')
class ContextualCiteSentIdentDatasetReader(DatasetReader):
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
                cur_sent_id = sent_json.get("cur_sent_id", "NULL").strip()
                section_name = sent_json.get("sec_name", "NULL").strip()

                prev_sent = sent_json.get("prev_sent", "NULL").strip()
                cur_sent = sent_json.get("cur_sent").strip()
                next_sent = sent_json.get("next_sent", "NULL").strip()

                prev_scaled_sent_len = sent_json.get("prev_scaled_len_features", {})
                cur_scaled_sent_len = sent_json.get("cur_scaled_len_features", {})
                next_scaled_sent_len = sent_json.get("next_scaled_len_features", {})

                prev_has_citation = sent_json.get("prev_has_citation", -1)
                next_has_citation = sent_json.get("next_has_citation", -1)

                cur_has_citation = sent_json.get("cur_has_citation")

                if (not section_name) or (not cur_sent) or (cur_has_citation is None):
                    print(line)
                    continue
                yield self.text_to_instance(
                    cur_sent_id,
                    section_name, prev_sent, cur_sent, next_sent, prev_scaled_sent_len, cur_scaled_sent_len,
                    next_scaled_sent_len, prev_has_citation, next_has_citation, cur_has_citation
                    )

    @overrides
    def text_to_instance(self,
                         cur_sent_id,
                         section_name,
                         prev_sent, cur_sent, next_sent,
                         prev_scaled_sent_len,
                         cur_scaled_sent_len,
                         next_scaled_sent_len,
                         prev_has_citation,
                         next_has_citation,
                         cur_has_citation
                         ) -> Instance:

        fields = {'label': LabelField(cur_has_citation, skip_indexing=True)}

        section_name_tokens = self._tokenizer.tokenize(section_name)
        fields["section_name"] = TextField(section_name_tokens, self._token_indexers)

        prev_sent_tokens = self._tokenizer.tokenize(prev_sent)
        fields["prev_sent"] = TextField(prev_sent_tokens, self._token_indexers)

        cur_sent_tokens = self._tokenizer.tokenize(cur_sent)
        fields["cur_sent"] = TextField(cur_sent_tokens, self._token_indexers)

        next_sent_tokens = self._tokenizer.tokenize(next_sent)
        fields["next_sent"] = TextField(cur_sent_tokens, self._token_indexers)

        prev_sent_len_vec = prev_scaled_sent_len.get("values", [0.0, 0.0])
        cur_sent_len_vec = cur_scaled_sent_len.get("values", [0.0, 0.0])
        next_sent_len_vec = next_scaled_sent_len.get("values", [0.0, 0.0])
        all_addi_features = prev_sent_len_vec + cur_sent_len_vec + next_sent_len_vec + [prev_has_citation,
                                                                                        next_has_citation]

        fields["additional_features"] = ArrayField(np.array(all_addi_features))

        fields['metadata'] = MetadataField(cur_sent_id)

        return Instance(fields)
