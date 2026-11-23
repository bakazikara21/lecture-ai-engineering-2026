"""第5回: ページ単位のベクトル検索。"""

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_NEIGHBOR_WINDOW, RAG_SOURCE, RAG_TOP_K, RAG_TREE,
    get_embedding_vertex_config,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
)
import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from collections import defaultdict, deque
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec
from dotenv import load_dotenv

load_dotenv()


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )


def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"]
                        if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update(
                {key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(
                    r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。

    agent.py の生成と同じく、API キーではなく Vertex AI（ADC 認証）で呼ぶ。
    索引（index.sqlite3）を作ったときと同じ EMBEDDING_MODEL・次元数で埋め込むこと。
    """
    project, location = get_embedding_vertex_config()
    with genai.Client(vertexai=True, project=project, location=location) as client:
        # Vertex AI では埋め込みモデルに count_tokens が使えない（400 になる）ので、事前には数えない。
        # 代わりに auto_truncate=False で、モデルの上限を超えた入力を黙って切り捨てずエラーにさせる。
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS,
                auto_truncate=False),
        )
    embedding = response.embeddings[0]
    stats = embedding.statistics
    if stats and stats.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    # Vertex AI は使ったトークン数を返すので、それで余裕分を残した上限（EMBEDDING_MAX_TOKENS）を確かめる
    if stats and stats.token_count is not None and stats.token_count > EMBEDDING_MAX_TOKENS:
        raise ValueError(f"埋め込み入力が上限を超えています: {int(stats.token_count)} tokens。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), RAG_TOP_K))
        return [dict(row) for row in rows]


def find_neighbors(hits: list[dict], pages: list[dict], window: int = 1) -> list[dict]:
    """検索で見つかったページ（hits）の前後 window ページを、同じ資料の中から探して返す。

    引数:
        hits: search() の結果。page_id だけを使う。
        pages: load_pages() の結果。資料ごとにページ順で並んでいることを前提にする。
        window: 前後それぞれ何ページ付けるか。0 なら何も付けない。
    戻り値:
        前後ページのリスト。各要素は load_pages() のページ情報に、
        どのヒットの前後として付けたかを示す neighbor_of（ヒットの page_id）を足したもの。

    - 前後は「PDFのページ番号 ±window」ではなく「資料内の並び順」で決める。
      番号が飛んでいる資料でも隣のページを取れ、資料ごとに分けて数えるので別の資料にもまたがらない。
    - hits に含まれるページは前後ページとして返さない（本文の重複を避ける）。
    - 複数のヒットに共通する前後ページは1回だけ返し、検索順位が上のヒットの前後として扱う。
    """
    if window < 0:
        raise ValueError("前後ページ数は0以上にしてください。")
    by_document = defaultdict(list)
    for page in pages:
        by_document[page["document_id"]].append(page)
    # page_id → (資料ID, 資料内での位置)。ヒットの位置を引くための表
    position = {page["page_id"]: (document_id, index)
                for document_id, document_pages in by_document.items()
                for index, page in enumerate(document_pages)}

    # ヒット自身は最初から「出した」ことにしておき、前後ページとして二重に返さない
    seen = {hit["page_id"] for hit in hits}
    neighbors = []
    for hit in hits:  # 検索順位の順に処理するので、共通の前後ページは上位のヒットに付く
        if hit["page_id"] not in position:
            continue  # 対象資料外のページ（索引と教材JSONがずれた場合など）は前後を探さない
        document_id, index = position[hit["page_id"]]
        document_pages = by_document[document_id]
        # 前→後の順に並べる。range の端を資料の先頭・末尾で切るので、資料の外には出ない
        for i in range(max(0, index - window), min(len(document_pages), index + window + 1)):
            page = document_pages[i]
            if page["page_id"] in seen:
                # 重複するならcontinue?
                continue
            seen.add(page["page_id"])
            neighbors.append({**page, "neighbor_of": hit["page_id"]})
    return neighbors


def search_with_neighbors(query: str) -> dict:
    """ベクトル検索の結果に、前後 RAG_NEIGHBOR_WINDOW ページの本文を付けて返す。

    戻り値は {"results": search() の結果そのまま, "neighbors": 前後ページ}。
    search() 自体は変えず、別のキーで前後ページを返す。search() は search_graph() の入口にも
    使っているので、search() に前後を混ぜると Graph の結果まで変わってしまうため。
    """
    hits = search(query)
    return {"results": hits,
            "neighbors": find_neighbors(hits, load_pages(RAG_SOURCE), RAG_NEIGHBOR_WINDOW)}


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(
        GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current],
                               key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max(
                    (len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}


def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(
        materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    found = search_with_neighbors(args.query)
    # 検索で見つかったページと、前後として付けたページを見出しで区別して表示する
    labeled = [(f"検索結果 {i}", result)
               for i, result in enumerate(found["results"], 1)]
    labeled += [(f"前後ページ（{result['neighbor_of']} の前後）", result)
                for result in found["neighbors"]]
    for label, result in labeled:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
