{
  "dataset_reader": {
    "name": "basic_classification_reader",
    "data_path": "/home/lsm/projects/CF_question_classifier/question_classifier_deeppavlov/data"
  },
  "dataset_iterator": {
    "name": "basic_classification_iterator",
    "seed": 42,
    "fields_to_merge": [
      "train",
      "valid"
    ],
    "merged_field": "train",
    "field_to_split": "train",
    "split_fields": [
      "train",
      "valid"
    ],
    "split_proportions": [
      0.9,
      0.1
    ]
  },
  "chainer": {
    "pipe": [
          {
          "in":[
          "y"
          ],
        "id": "label_transformer",
        "name": "label_transformer",
        "nclasses":19,
        "fit_on": ["y"],
        "out":[
        "yoh"
        ]
      },
      {
        "name": "text_normalizer",
        "id": "text_normalizer",
        "in": [
          "x"
        ],
        "out": [
          "xn"
        ]
      },
      {
        "name": "embedder",
        "in": [
          "xn"
        ],
        "out": [
          "xv"
        ]
      },

      {
        "name": "cnn_model",
        "in": [
          "xv"
        ],
        "in_y":[
          "yoh"
        ],
        "out": [
          "y_pred"
        ],
        "loss": "binary_crossentropy",
        "metrics": "accuracy",
        "optimizer": "adam",
        "opt": {
          "cnn_layers": [
            {
              "filters": 42,
              "kernel_size": 2
            },
            {
              "filters": 20,
              "kernel_size": 2
            },
            {
              "filters": 15,
              "kernel_size": 2
            },
            {
              "filters": 10,
              "kernel_size": 2
            }
          ],
          "emb_dim": 50,
          "seq_len": 50,
          "pool_size": 4,
          "dropout_power": 0.5,
          "n_classes": 5,
          "pooling_size":1,
          "classes":[
          "класс1",
          "класс2",
          "класс3",
          "класс4",
          "класс5"
          ]
        }
        
      }
    ],
    "out": [
      "y_pred"
    ],
    "in": [
      "x"
    ],
    "in_y": [
      "y"
    ]
  },
  "train": {
    "epochs": 150,
    "batch_size": 100,
    "metrics": [
      "sets_accuracy"
    ],
    "val_every_n_epochs": 5,
    "log_every_n_epochs": 10
  }
}
