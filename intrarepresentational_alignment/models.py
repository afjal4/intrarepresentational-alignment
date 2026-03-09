from enum import StrEnum


class EmbeddingModel(StrEnum):
    # Sentence-Transformers family
    ALL_MINILM_L6_V2        = "sentence-transformers/all-MiniLM-L6-v2"
    ALL_MINILM_L12_V2       = "sentence-transformers/all-MiniLM-L12-v2"
    ALL_MPNET_BASE_V2       = "sentence-transformers/all-mpnet-base-v2"
    PARAPHRASE_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # BAAI/bge family
    BGE_SMALL_EN            = "BAAI/bge-small-en-v1.5"
    BGE_BASE_EN             = "BAAI/bge-base-en-v1.5"
    BGE_LARGE_EN            = "BAAI/bge-large-en-v1.5"
    BGE_M3                  = "BAAI/bge-m3"

    # intfloat/e5 family
    E5_SMALL_V2             = "intfloat/e5-small-v2"
    E5_BASE_V2              = "intfloat/e5-base-v2"
    E5_LARGE_V2             = "intfloat/e5-large-v2"
    E5_MISTRAL_7B           = "intfloat/e5-mistral-7b-instruct"

    # GTE family
    GTE_SMALL               = "thenlper/gte-small"
    GTE_LARGE               = "thenlper/gte-large"

    # Nomic
    NOMIC_EMBED_TEXT_V1     = "nomic-ai/nomic-embed-text-v1"
