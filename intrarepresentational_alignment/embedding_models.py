from enum import StrEnum


class EmbeddingModel(StrEnum):
    # word2vec (Google News)
    WORD2VEC_GOOGLE_NEWS_300     = "word2vec-google-news-300"

    # GloVe (Wikipedia + Gigaword)
    GLOVE_WIKI_GIGAWORD_50       = "glove-wiki-gigaword-50"
    GLOVE_WIKI_GIGAWORD_100      = "glove-wiki-gigaword-100"
    GLOVE_WIKI_GIGAWORD_300      = "glove-wiki-gigaword-300"

    # FastText (Wikipedia news, with subword)
    FASTTEXT_WIKI_NEWS_300       = "fasttext-wiki-news-subwords-300"

    # ConceptNet Numberbatch (commonsense knowledge graph embeddings)
    CONCEPTNET_NUMBERBATCH_300   = "conceptnet-numberbatch-en-19.08"
