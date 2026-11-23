"""講座全体で共通の設定。

APIキーやモデル名はコードに書かず、環境変数から読み込む。
"""

import os
from pathlib import Path

# 既定モデル。環境変数 AIE_MODEL を設定すると切り替えられる。
#   例: export AIE_MODEL=gemini-3.5-flash
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL = os.getenv("AIE_MODEL", DEFAULT_MODEL)

# 第5回: 1ページを1チャンクとして検索する。索引はscripts/build_rag_index.pyで作成する。
ROOT = Path(__file__).resolve().parent.parent
RAG_SOURCE = ROOT / "data/rag_data/lectures.json"
RAG_TREE = ROOT / "data/rag_data/tree.json"
RAG_GRAPH = ROOT / "data/rag_data/graph.json"
RAG_DB = Path(os.getenv("AIE_RAG_DB", str(
    ROOT / "data/rag_data/index.sqlite3")))
RAG_DOCUMENT_IDS = ("lec01-main", "lec02-main", "lec03-main")
RAG_TOP_K = 5
# 第5回宿題（実装）: search_materials で、検索で見つかったページの前後何ページの本文も付けるか。
# 0 にすると前後を付けない（改良前と同じ取得範囲）。比較のため環境変数 AIE_RAG_NEIGHBOR_WINDOW で切り替える。
RAG_NEIGHBOR_WINDOW = int(os.getenv("AIE_RAG_NEIGHBOR_WINDOW", "1"))
GRAPH_START_K = 3
GRAPH_HOPS = 2
GRAPH_MAX_RESULTS = 10
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
# 2,048トークンの上限に対し、タスク指定等の分を残す。超過時は切り捨てず停止する。
EMBEDDING_MAX_TOKENS = 2000


def api_error_message(status: int, detail: str = "") -> str:
    """HTTPステータスコードを、受講生向けの日本語の説明にする（ask.py と agent.py で共用）。"""
    if status == 429:
        return "レート制限に達しました。1分ほど待ってから再実行してください。"
    if status == 503:
        return "モデルが混み合っています（503）。少し待ってから再実行してください。"
    if status in (400, 401, 403, 404):
        return f"APIキーかモデル名に問題があります（{status}）: {detail}"
    return f"APIエラー（{status}）: {detail}"


def get_vertex_config() -> tuple[str, str]:
    """Vertex AI の接続先（プロジェクトID, リージョン）を環境変数から読んで返す。

    agent.py（生成）と retrieval.py（埋め込み）の両方が同じ接続先を使うので、ここで1か所にまとめる。
    認証は API キーではなく ADC（gcloud auth application-default login のログイン情報）で行う。
    値は .env に書き、コードには直書きしない。未設定なら原因が分かるメッセージで終了する。
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    if not project or not location:
        raise SystemExit(
            ".env に GOOGLE_CLOUD_PROJECT と GOOGLE_CLOUD_LOCATION を書いてください")
    return project, location


# gemini-embedding-001 は Vertex AI の "global" では提供されない（404 になる）。
# 生成モデルは global で呼べるので、埋め込みだけ地域のリージョンで呼ぶ。
EMBEDDING_FALLBACK_LOCATION = "us-central1"


def get_embedding_vertex_config() -> tuple[str, str]:
    """埋め込み用の Vertex AI の接続先（プロジェクトID, リージョン）を返す。

    環境変数 AIE_EMBEDDING_LOCATION があればそれを使う。なければ GOOGLE_CLOUD_LOCATION を使い、
    それが "global" のときだけ EMBEDDING_FALLBACK_LOCATION に置き換える。
    """
    project, location = get_vertex_config()
    override = os.getenv("AIE_EMBEDDING_LOCATION")
    if override:
        return project, override
    return project, EMBEDDING_FALLBACK_LOCATION if location == "global" else location


def get_api_key() -> str:
    """環境変数 GEMINI_API_KEY を返す。未設定なら手順を表示して終了する。"""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "環境変数 GEMINI_API_KEY が見つかりません。\n"
            "1. GitHub の Settings → Codespaces → Secrets に GEMINI_API_KEY を登録し、\n"
            "   使う repository に自分の fork を選んでください。\n"
            "2. Codespaces を起動したあとに登録した場合は、Codespaces を再起動してください。\n"
            "   （左下の「Codespaces」→「Stop Current Codespace」→ 再度開く）"
        )
    return key
