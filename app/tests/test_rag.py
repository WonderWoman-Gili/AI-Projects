from app.rag import embed, search

def test_embedding_dimension():
    vec = embed("hello world")
    assert len(vec) == 768
    
    