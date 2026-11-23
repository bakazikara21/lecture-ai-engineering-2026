"""retrieval.py のテスト。実際の API は呼ばない。

- 埋め込みが Vertex AI（ADC 認証）で接続すること：genai.Client を偽物に差し替えて、渡された接続先と設定を確かめる
- 検索結果に前後ページを付ける処理（find_neighbors / search_with_neighbors）：小さな偽の資料で確かめる
"""

import json
from types import SimpleNamespace

import pytest

from app import retrieval
from app.config import get_embedding_vertex_config, get_vertex_config


class FakeClient:
    """genai.Client の代わり。受け取った引数を記録し、決まった埋め込みを返す。"""

    created: list[dict] = []
    configs: list = []
    token_count = 3

    def __init__(self, **kwargs):
        FakeClient.created.append(kwargs)
        self.models = SimpleNamespace(embed_content=self._embed)

    def _embed(self, **kwargs):
        FakeClient.configs.append(kwargs["config"])
        stats = SimpleNamespace(truncated=False, token_count=FakeClient.token_count)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0], statistics=stats)])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.delenv("AIE_EMBEDDING_LOCATION", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # API キーが無くても動くこと
    FakeClient.created, FakeClient.configs, FakeClient.token_count = [], [], 3
    monkeypatch.setattr(retrieval.genai, "Client", FakeClient)
    return FakeClient


def test_embed_text_uses_vertex_ai(fake_client):
    vector = retrieval.embed_text("LoRAとは？", "RETRIEVAL_QUERY")

    # global では埋め込みモデルが無いので、地域のリージョンに置き換わる
    assert fake_client.created == [
        {"vertexai": True, "project": "test-project", "location": "us-central1"}]
    assert fake_client.configs[0].auto_truncate is False  # 黙って切り捨てさせない
    assert vector == pytest.approx([0.6, 0.8])  # 長さ1に正規化される


def test_embed_text_rejects_too_many_tokens(fake_client):
    fake_client.token_count = retrieval.EMBEDDING_MAX_TOKENS + 1
    with pytest.raises(ValueError, match="上限"):
        retrieval.embed_text("長い本文", "RETRIEVAL_DOCUMENT")


def test_embedding_location_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
    monkeypatch.delenv("AIE_EMBEDDING_LOCATION", raising=False)
    assert get_embedding_vertex_config() == ("test-project", "asia-northeast1")
    monkeypatch.setenv("AIE_EMBEDDING_LOCATION", "europe-west4")
    assert get_embedding_vertex_config() == ("test-project", "europe-west4")


def test_vertex_config_missing_stops_with_message(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    with pytest.raises(SystemExit, match="GOOGLE_CLOUD_PROJECT"):
        get_vertex_config()


# ---- 検索結果に前後ページを付ける処理 ----

def make_pages() -> list[dict]:
    """資料A（3ページ）と資料B（2ページ）を、load_pages() と同じ形・順で作る。

    資料Aの末尾と資料Bの先頭は並びの上では隣り合うので、資料をまたがないことを確かめられる。
    資料Aはページ番号をわざと飛ばし（1, 2, 5）、番号ではなく並び順で前後を決めていることも確かめる。
    """
    layout = {"A": [1, 2, 5], "B": [1, 2]}
    return [{"page_id": f"{doc}-p{n}", "document_id": doc, "page": n, "content": f"{doc}{n}の本文"}
            for doc, numbers in layout.items() for n in numbers]


def neighbor_ids(hit_ids: list[str], window: int = 1) -> list[str]:
    hits = [{"page_id": page_id} for page_id in hit_ids]
    return [page["page_id"] for page in retrieval.find_neighbors(hits, make_pages(), window)]


def test_neighbors_are_previous_then_next():
    assert neighbor_ids(["A-p2"]) == ["A-p1", "A-p5"]


def test_neighbors_stop_at_document_edges():
    assert neighbor_ids(["A-p1"]) == ["A-p2"]  # 先頭ページには前がない
    assert neighbor_ids(["A-p5"]) == ["A-p2"]  # 資料Aの末尾。並びで隣の資料Bの先頭には進まない
    assert neighbor_ids(["B-p1"]) == ["B-p2"]  # 資料Bの先頭。資料Aの末尾には戻らない


def test_hits_are_not_repeated_as_neighbors():
    # A-p1 と A-p2 はどちらもヒット。互いの前後としては返さず、残りの A-p5 だけを付ける
    assert neighbor_ids(["A-p1", "A-p2"]) == ["A-p5"]


def test_shared_neighbor_is_returned_once_for_higher_ranked_hit():
    neighbors = retrieval.find_neighbors(
        [{"page_id": "A-p5"}, {"page_id": "A-p1"}], make_pages())
    assert [page["page_id"] for page in neighbors] == ["A-p2"]
    assert neighbors[0]["neighbor_of"] == "A-p5"  # 検索順位が上のヒットの前後として扱う


def test_neighbor_keeps_page_information():
    [neighbor] = retrieval.find_neighbors([{"page_id": "B-p2"}], make_pages())
    assert neighbor == {"page_id": "B-p1", "document_id": "B", "page": 1,
                        "content": "B1の本文", "neighbor_of": "B-p2"}


def test_window_controls_range():
    assert neighbor_ids(["A-p2"], window=0) == []  # 0 なら改良前と同じく前後を付けない
    assert neighbor_ids(["A-p1"], window=2) == ["A-p2", "A-p5"]
    with pytest.raises(ValueError, match="0以上"):
        neighbor_ids(["A-p1"], window=-1)


def test_unknown_hit_is_skipped():
    assert neighbor_ids(["Z-p9", "B-p2"]) == ["B-p1"]


def test_search_with_neighbors_keeps_search_results(monkeypatch):
    hits = [{"page_id": "A-p2", "distance": 0.1}]
    monkeypatch.setattr(retrieval, "search", lambda query: hits)
    monkeypatch.setattr(retrieval, "load_pages", lambda source: make_pages())
    monkeypatch.setattr(retrieval, "RAG_NEIGHBOR_WINDOW", 1)

    found = retrieval.search_with_neighbors("質問")
    assert found["results"] == hits  # 検索結果そのものは変えない（search_graph の入口と同じ）
    assert [page["page_id"] for page in found["neighbors"]] == ["A-p1", "A-p5"]


def test_search_materials_returns_neighbors(monkeypatch):
    from app import tools

    monkeypatch.setattr(retrieval, "search", lambda query: [{"page_id": "B-p1"}])
    monkeypatch.setattr(retrieval, "load_pages", lambda source: make_pages())
    monkeypatch.setattr(retrieval, "RAG_NEIGHBOR_WINDOW", 1)

    payload = json.loads(tools.search_materials("質問"))
    assert payload["results"] == [{"page_id": "B-p1"}]
    assert [page["page_id"] for page in payload["neighbors"]] == ["B-p2"]
