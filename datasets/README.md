# Datasets folder

## Getting started

1. Download both the [large and small MIND datasets](https://msnews.github.io/).  I placed them in folders called `MINDlarge` and `MINDsmall`.

## Project File Structure

The `cicero` project has a structured directory for datasets and subdirectories. Here's the layout for the large MIND dataset.  It's similar with the small and demo datasets.

```plaintext
## Project File Structure

The `cicero` project has a structured directory for datasets and subdirectories. Here's an example of directory and file layout:

```plaintext
cicero/
│
├── datasets/                          # Root folder for datasets
│   ├── MINDlarge/                     # Large dataset
│   │   ├── dev/                       # Development dataset
│   │   │   ├── behaviors.tsv          # User behavior data for development
│   │   │   ├── entity_embeddings.vec  # Entity embeddings for development
│   │   │   ├── news.tsv               # News article data for development
│   │   │   └── relation_embedding.vec # Relation embeddings for development
│   │   │
│   │   ├── test/                      # Test dataset
│   │   │   ├── behaviors.tsv          # User behavior data for testing
│   │   │   ├── entity_embeddings.vec  # Entity embeddings for testing
│   │   │   ├── news.tsv               # News article data for testing
│   │   │   └── relation_embedding.vec # Relation embeddings for testing
│   │   │
│   │   └── train/                     # Training dataset
│   │       ├── behaviors.tsv          # User behavior data for training
│   │       ├── entity_embeddings.vec  # Entity embeddings for training
│   │       ├── news.tsv               # News article data for training
│   │       └── relation_embedding.vec # Relation embeddings for training
│   │
│   │   └── utils/                     # Utilities for MINDlarge
│   │       ├── embedding.npy          # Precomputed embedding matrix
│   │       ├── embedding_all.npy      # Precomputed embedding matrix for all words
│   │       ├── lstur.yaml             # Configuration for LSTUR model
│   │       ├── naml.yaml              # Configuration for NAML model
│   │       ├── npa.yaml               # Configuration for NPA model
│   │       ├── nrms.yaml              # Configuration for NRMS model
│   │       ├── uid2index.pkl          # User ID to index mapping
│   │       ├── word_dict.pkl          # Word to index dictionary
│   │       └── word_dict_all.pkl      # Word to index dictionary for all words
│   │
│   ├── MINDsmall/                     # Small dataset
│   │   ├── dev/                       # Development dataset
│   │   │   ├── behaviors.tsv          # User behavior data for development
│   │   │   ├── entity_embeddings.vec  # Entity embeddings for development
│   │   │   ├── news.tsv               # News article data for development
│   │   │   └── relation_embedding.vec # Relation embeddings for development
│   │   │
│   │   ├── test/                      # Test dataset
│   │   │   ├── behaviors.tsv          # User behavior data for testing
│   │   │   ├── entity_embeddings.vec  # Entity embeddings for testing
│   │   │   ├── news.tsv               # News article data for testing
│   │   │   └── relation_embedding.vec # Relation embeddings for testing
│   │   │
│   │   └── train/                     # Training dataset
│   │       ├── behaviors.tsv          # User behavior data for training
│   │       ├── entity_embeddings.vec  # Entity embeddings for training
│   │       ├── news.tsv               # News article data for training
│   │       └── relation_embedding.vec # Relation embeddings for training
│   │
│   │   └── utils/                     # Utilities for MINDsmall
│   │       ├── embedding.npy          # Precomputed embedding matrix
│   │       ├── embedding_all.npy      # Precomputed embedding matrix for all words
│   │       ├── lstur.yaml             # Configuration for LSTUR model
│   │       ├── naml.yaml              # Configuration for NAML model
│   │       ├── npa.yaml               # Configuration for NPA model
│   │       ├── nrms.yaml              # Configuration for NRMS model
│   │       ├── uid2index.pkl          # User ID to index mapping
│   │       ├── word_dict.pkl          # Word to index dictionary
│   │       └── word_dict_all.pkl      # Word to index dictionary for all words
│   │
│   └── MINDdemo/                      # Demo dataset
│       ├── dev/                       # Development dataset
│       │   ├── behaviors.tsv          # User behavior data for development
│       │   ├── entity_embeddings.vec  # Entity embeddings for development
│       │   ├── news.tsv               # News article data for development
│       │   └── relation_embedding.vec # Relation embeddings for development
│       │
│       ├── test/                      # Test dataset
│       │   ├── behaviors.tsv          # User behavior data for testing
│       │   ├── entity_embeddings.vec  # Entity embeddings for testing
│       │   ├── news.tsv               # News article data for testing
│       │   └── relation_embedding.vec # Relation embeddings for testing
│       │
│       └── train/                     # Training dataset
│       │   ├── behaviors.tsv          # User behavior data for training
│       │   ├── entity_embeddings.vec  # Entity embeddings for training
│       │   ├── news.tsv               # News article data for training
│       │   └── relation_embedding.vec # Relation embeddings for training
│       │
│       └── utils/                     # Utilities for MINDdemo
│           ├── embedding.npy          # Precomputed embedding matrix
│           ├── embedding_all.npy      # Precomputed embedding matrix for all words
│           ├── lstur.yaml             # Configuration for LSTUR model
│           ├── naml.yaml              # Configuration for NAML model
│           ├── npa.yaml               # Configuration for NPA model
│           ├── nrms.yaml              # Configuration for NRMS model
│           ├── subvert_dict.pkl       # Sub-vertical mapping dictionary
│           ├── uid2index.pkl          # User ID to index mapping
│           ├── vert_dict.pkl          # Vertical mapping dictionary
│           ├── word_dict.pkl          # Word to index dictionary
│           └── word_dict_all.pkl      # Word to index dictionary for all words
│
└── README.md                          # Instructions for the project



```