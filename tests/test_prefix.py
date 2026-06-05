from scripts.rag_qdrant.prefix import parse_prefix


def test_embed_with_space():
    mode, body = parse_prefix("embed hello world")
    assert mode == "embed"
    assert body == "hello world"


def test_embed_with_multiple_spaces():
    mode, body = parse_prefix("embed   hello world")
    assert mode == "embed"
    assert body == "hello world"


def test_embed_with_tab():
    mode, body = parse_prefix("embed\thello world")
    assert mode == "embed"
    assert body == "hello world"


def test_embed_with_newline():
    mode, body = parse_prefix("embed\nhello world")
    assert mode == "embed"
    assert body == "hello world"


def test_embed_capitalized():
    mode, body = parse_prefix("Embed hello world")
    assert mode == "embed"
    assert body == "hello world"


def test_embed_uppercase():
    mode, body = parse_prefix("EMBED hello world")
    assert mode == "embed"
    assert body == "hello world"


def test_embed_mixed_case():
    mode, body = parse_prefix("eMbEd hello world")
    assert mode == "embed"
    assert body == "hello world"


def test_embed_alone():
    mode, body = parse_prefix("embed")
    assert mode == "embed"
    assert body == ""


def test_embed_alone_with_space():
    mode, body = parse_prefix("embed ")
    assert mode == "embed"
    assert body == ""


def test_embed_at_end_is_not_embed():
    mode, body = parse_prefix("hello embed world")
    assert mode == "query"
    assert body == "hello embed world"


def test_embedding_prefix_is_not_embed():
    mode, body = parse_prefix("embedding hello")
    assert mode == "query"
    assert body == "embedding hello"


def test_embedded_word_is_not_embed():
    mode, body = parse_prefix("I am embedded in text")
    assert mode == "query"
    assert body == "I am embedded in text"


def test_empty_string():
    mode, body = parse_prefix("")
    assert mode == "query"
    assert body == ""


def test_plain_question():
    mode, body = parse_prefix("What is the capital of France?")
    assert mode == "query"
    assert body == "What is the capital of France?"


def test_embed_word_boundary_punctuation():
    mode, body = parse_prefix("embed! hello world")
    assert mode == "query"
    assert body == "embed! hello world"


def test_embed_then_body():
    mode, body = parse_prefix("embed\n\nThe quick brown fox.")
    assert mode == "embed"
    assert body == "The quick brown fox."
