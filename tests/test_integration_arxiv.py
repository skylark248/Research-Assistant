import pytest

pytestmark = pytest.mark.integration


def test_real_search_and_lookup():
    from rag.arxiv_client import get_paper, search_papers

    papers = search_papers("attention is all you need", max_results=3)
    assert papers and all(p.paper_id and p.title for p in papers)

    meta = get_paper("1706.03762")
    assert meta is not None
    assert "Attention" in meta.title
