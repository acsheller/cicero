# Datasets folder

## Getting started

1. Download both the [large and small MIND datasets](https://msnews.github.io/).  I placed them in folders called `MINDlarge` and `MINDsmall`.

## Project File Structure

The `cicero` project has a structured directory for datasets and subdirectories. Here's the layout for the large MIND dataset.  It's similar with the small and demo datasets.

```plaintext
## Project File Structure

The `cicero` project has a structured directory for datasets and subdirectories. Here's the detailed layout:

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
│           ├── behaviors.tsv          # User behavior data for training
│           ├── entity_embeddings.vec  # Entity embeddings for training
│           ├── news.tsv               # News article data for training
│           └── relation_embedding.vec # Relation embeddings for training
│
└── README.md                          # Instructions for the project

```
