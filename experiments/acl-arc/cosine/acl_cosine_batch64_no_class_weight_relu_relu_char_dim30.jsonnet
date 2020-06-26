local HOME = "/home/xxxxxx/";

{
  "dataset_reader": {
    "type": "acl_sent_reader",
    "lazy": false,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters"
      }
    }
  },
  "train_data_path": HOME + "acl_arc/training.jsonl",
  "validation_data_path": HOME +"acl_arc/validation.jsonl",
  "test_data_path": HOME +"acl_arc/testing.jsonl",
  "evaluate_on_test": true,
  "model": {
    "type": "sent_att",
    "label_namespace": "labels",
    "dropout":0.5,
    "class_weight":[1, 1],
    "calculate_f1":true,
    "calculate_auc":true,
    "calculate_auc_pr":true,
    "positive_label":1,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
          "trainable": true
        },
        "token_characters": {
          "type": "character_encoding",
           "embedding": {
             "embedding_dim": 10,
             "num_embeddings": 262
          },
          "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.25,
            "hidden_size": 15,
            "input_size": 10,
            "num_layers": 2
          }
        }
      }
    },
    "sent_encoder": {
      "type": "lstm",
      "input_size": 330,
      "hidden_size": 128,
      "num_layers": 1,
      "dropout": 0.5,
      "bidirectional": true
    },
    "classifier_feedforward": {
      "input_dim": 128*2*1+2,
      "num_layers": 2,
      "hidden_dims": [64, 2],
      "activations": ["relu","relu"],
      "dropout": [0.2,0.0]
    },
    "encoder_attention":{
      "type":"cosine",
      "normalize":true
    },
    "regularizer": [
        ["weight$", {"type": "l2", "alpha": 0.0000001}],
        ["bias$", {"type": "l2", "alpha": 0.0000001}]
    ]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["cur_sent", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 64,
    "cache_instances": true
    //"biggest_batch_first": true
  },

  "trainer": {
    "num_epochs": 100,
    "patience": 3,
    "cuda_device": -1,
    "grad_clipping": 10.0,
    "validation_metric": "+f1",
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-07
    }
  }
}
