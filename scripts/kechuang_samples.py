"""Extract an author's rendered posts from local Kechuang MHTML snapshots.

Uses only the standard library. Author labels are declarations recorded before
inference; quoted text, media controls, and page metadata never enter the corpus.
The resulting thread samples can use public_samples.py's existing detect action.
"""
import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TZ = timezone(timedelta(hours=8))
VOID = set("area base br col embed hr img input link meta param source track wbr".split())
SKIP = set("script style button input select option textarea iframe svg noscript".split())
BLOCK = set("p div section article h1 h2 h3 h4 h5 h6 ul ol table tr pre".split())
CSS = """body{max-width:1000px;margin:32px auto;padding:0 24px;font:17px/1.8 system-ui,sans-serif;color:#252525;background:#fcfcfa}a{color:#126397}h1{font-size:28px;line-height:1.4}h2{font-size:22px}article{border-top:1px solid #ccc;margin-top:30px;padding-top:12px}.meta,.media{font-size:14px;color:#666}p{margin:12px 0}img{max-width:100%;height:auto}figure{margin:16px 0}code{white-space:pre-wrap;font-size:14px;background:#f0f0ec}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #ddd;text-align:left}nav{padding:12px;background:#eee}th{background:#eee}"""


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def document(title, body):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\' data:; style-src \'unsafe-inline\'">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style><body>{body}</body></html>')


class Node:
    def __init__(self, tag="", attrs=(), parent=None, start=0, inner_start=0):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.start, self.inner_start, self.end = start, inner_start, inner_start
        self.children = []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self):
        return "".join(c.text() if isinstance(c, Node) else c for c in self.children)


class Tree(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.lines = [0] + [m.end() for m in re.finditer("\n", source)]
        self.root = Node()
        self.stack = [self.root]
        self.feed(source)
        self.close()

    def source_position(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def handle_starttag(self, tag, attrs):
        start = self.source_position()
        node = Node(tag, attrs, self.stack[-1], start, start + len(self.get_starttag_text()))
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                for node in self.stack[index:]:
                    node.end = self.source_position()
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def clean(value):
    value = value.replace("\xa0", " ").replace("\r", "")
    # Blink/editor direction markers around media are not authored prose.
    value = value.translate(dict.fromkeys(map(ord, "\u200b\u200e\u200f\ufeff\u2060")))
    value = "\n".join(line.strip() for line in value.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def is_post(node):
    return node.tag == "div" and "data-pid" in node.attrs and "data-post-hide" in node.attrs


def owner(body):
    node = body
    while node.parent and not is_post(node):
        node = node.parent
    # The main post has a separate article card instead of a reply data-pid.
    container = node if is_post(node) else body.parent.parent
    anchors = [n for n in container.walk() if n.tag == "a"
               and n.attrs.get("data-global-mouseover") == "showUserPanel"
               and re.search(r"/u/\d+$", n.attrs.get("href", ""))]
    if not anchors:
        raise ValueError("Rendered body has no identifiable author header")
    return anchors[0], container


def date_titles(container, before):
    return list(dict.fromkeys(n.attrs["title"] for n in container.walk()
                if n.start < before and re.fullmatch(r"\d{4}/\d\d/\d\d \d\d:\d\d:\d\d", n.attrs.get("title", ""))))


class Renderer:
    def __init__(self, parts, directory, assets):
        self.parts, self.directory, self.assets = parts, directory, assets
        self.media = []
        self.quotes = []
        self.math_count = 0

    def image(self, node):
        source = node.attrs.get("src", "")
        part = self.parts.get(source)
        record = {"type": "image", "source": source, "available_in_snapshot": part is not None}
        if part is not None:
            mime = part.get_content_type()
            extensions = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                          "image/webp": ".webp", "image/svg+xml": ".svg", "image/avif": ".avif"}
            if mime not in extensions:
                raise ValueError(f"Unsupported image part: {mime}")
            raw = part.get_payload(decode=True)
            sha = digest(raw)
            rel = "assets/" + sha + extensions[mime]
            if rel not in self.assets:
                (self.directory / rel).write_bytes(raw)
                self.assets[rel] = {"path": rel, "sha256": sha, "bytes": len(raw), "mime_type": mime}
            record.update(file=rel, sha256=sha)
            rendered = f'<figure><img loading="lazy" src="../{rel}" alt="帖子原图"></figure>'
        else:
            rendered = '<p class="media">[图片：当前 MHTML 未包含图片文件]</p>'
        self.media.append(record)
        return "", rendered

    def render(self, node, pre=False):
        if isinstance(node, str):
            value = node.replace("\xa0", " ")
            if not pre:
                if not value.strip() and ("\n" in value or "\r" in value):
                    return "", ""
                value = re.sub(r"[\t\r\n ]+", " ", value)
            return value, html.escape(value)
        tag, attrs = node.tag, node.attrs
        if tag == "blockquote":
            self.quotes.append({"characters_before_cleanup": len(node.text()), "raw_html_sha256": None})
            return "\n\n", ""
        if tag in SKIP:
            return "", ""
        if tag == "mjx-container":
            maths = [n for n in node.walk() if n.tag == "mjx-math" and "data-latex" in n.attrs]
            if not maths:
                raise ValueError("MathJax formula lacks recoverable LaTeX")
            latex = maths[0].attrs["data-latex"]
            self.math_count += 1
            text = ("\\[" + latex + "\\]") if attrs.get("display") == "true" else ("\\(" + latex + "\\)")
            return text, '<code class="math">' + html.escape(text) + "</code>"
        if attrs.get("data-tag") == "nkcsource":
            kind = attrs.get("data-type")
            if kind in ("video", "audio", "attachment"):
                names = [n.text().strip() for n in node.walk() if n.tag == "a" and n.text().strip()]
                self.media.append({"type": kind, "resource_id": attrs.get("data-id"), "names": names,
                                   "included_in_detection": False})
                label = {"video": "视频", "audio": "音频", "attachment": "附件"}[kind]
                return "", f'<p class="media">[{label} #{html.escape(attrs.get("data-id", ""))}：未参与文本检测]</p>'
            if kind == "sticker":
                self.media.append({"type": kind, "resource_id": attrs.get("data-id"), "included_in_detection": False})
                return "", '<span class="media">[表情贴图]</span>'
        if tag == "img":
            if attrs.get("data-type") == "twemoji":
                text = attrs.get("alt", "")
                if re.fullmatch(r"[0-9a-fA-F]{4,6}(?:-[0-9a-fA-F]{4,6})*", text):
                    text = "".join(chr(int(codepoint, 16)) for codepoint in text.split("-"))
                return text, html.escape(text)
            return self.image(node)
        if tag == "br":
            return "\n", "<br>"
        values = [self.render(c, pre or tag == "pre") for c in node.children]
        text = "".join(t for t, _ in values)
        markup = "".join(h for _, h in values)
        if tag == "a":
            href = attrs.get("href", "")
            if urlparse(href).scheme in ("https", "http"):
                markup = f'<a href="{html.escape(href, quote=True)}" rel="noreferrer">{markup}</a>'
        elif tag in {"b", "strong", "i", "em", "sup", "sub", "code", "pre", "p", "ul", "ol", "li",
                     "table", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}:
            markup = f"<{tag}>{markup}</{tag}>"
        if tag == "li":
            text = "\n" + text + "\n"
        elif tag in BLOCK:
            text = "\n\n" + text + "\n\n"
        elif tag in ("td", "th"):
            text += "\t"
        return text, markup


def collect(source_dir, directory, username):
    if directory.exists():
        raise FileExistsError("Choose a new output directory; existing collections are never overwritten")
    inputs = sorted(source_dir.glob("*.mhtml"))
    if not inputs:
        raise ValueError("No MHTML inputs found")
    # Complete author selection and extraction before loading any detector.
    directory.mkdir(parents=True)
    for name in ("texts", "post-texts", "readable", "assets", "sources"):
        (directory / name).mkdir()
    now = datetime.now(LOCAL_TZ).isoformat()
    records, posts, evidence, assets, index_rows = [], [], [], {}, []
    author_ids = set()
    for source in inputs:
        raw = source.read_bytes()
        message = BytesParser(policy=policy.default).parsebytes(raw)
        url = message.get("Snapshot-Content-Location")
        parts = list(message.walk())
        candidates = [p for p in parts if p.get_content_type() == "text/html"]
        main = next((p for p in candidates if p.get("Content-Location") == url), candidates[0])
        payload = main.get_payload(decode=True)
        # Blink omits the MIME charset here but declares UTF-8 in HTML's meta.
        charset = main.get_content_charset()
        if charset is None:
            match = re.search(br"charset\s*=\s*[\"']?([A-Za-z0-9_-]+)", payload[:4096], re.I)
            if match is None:
                raise ValueError("Main HTML has neither MIME nor meta charset")
            charset = match.group(1).decode("ascii")
        source_html = payload.decode(charset, errors="strict")
        tree = Tree(source_html)
        nodes = list(tree.root.walk())
        bodies = [n for n in nodes if n.attrs.get("data-type") == "nkc-render-content"]
        title_node = next(n for n in nodes if n.tag == "title")
        title = title_node.text().removesuffix(" - 科创").strip()
        thread_id = re.search(r"/t/(\d+)", url).group(1)
        sample_id = "kechuang-t" + thread_id
        parts_by_url = {}
        for part in parts:
            if part.get("Content-Location"):
                parts_by_url[part["Content-Location"]] = part
            if part.get("Content-ID"):
                parts_by_url["cid:" + part["Content-ID"].strip("<>")] = part
        notices = [n.text().strip() for n in nodes if "本文包含AI生成内容" in n.text()
                   and not any(isinstance(c, Node) and "本文包含AI生成内容" in c.text() for c in n.children)]
        audit, selected, post_html = [], [], []
        thread_text = ""
        for body in bodies:
            author, container = owner(body)
            author_name = author.text().strip()
            uid = author.attrs["href"].rsplit("/", 1)[-1]
            pid = container.attrs.get("data-pid")
            audit.append({"document_id": body.attrs.get("data-id"), "post_id": pid,
                          "author": author_name, "uid": uid, "selected": author_name == username})
            if author_name != username:
                continue
            author_ids.add(uid)
            post_id = sample_id + ("-p" + pid if pid else "-main")
            renderer = Renderer(parts_by_url, directory, assets)
            text, markup = renderer.render(body)
            text = clean(text)
            if "\ufffd" in text:
                raise ValueError("Replacement character in extracted text")
            quoted_nodes = [n for n in body.walk() if n.tag == "blockquote"]
            quotes = [{"raw_html_sha256": digest(source_html[n.start:n.end].encode("utf-8")),
                       "characters_before_cleanup": len(n.text())} for n in quoted_nodes]
            floors = [n.text().strip() for n in container.walk() if n.tag == "a"
                      and n.attrs.get("href") == "https://www.kechuang.org/p/" + str(pid)]
            floor = floors[0] if pid and floors else "主帖"
            dates = date_titles(container, body.start)
            if text:
                if thread_text:
                    thread_text += "\n\n"
                start = len(thread_text)
                thread_text += text
                end = len(thread_text)
            else:
                start = end = len(thread_text)
            post = {"id": post_id, "thread_sample_id": sample_id, "title": title,
                    "author": username, "uid": uid, "source_label": "human", "provenance": "author_declaration",
                    "source_url": "https://www.kechuang.org/p/" + pid if pid else url,
                    "post_id": pid, "document_id": body.attrs.get("data-id"), "floor": floor,
                    "published_at": dates[0] if dates else None, "header_dates": dates,
                    "text": text, "characters": len(text), "sha256": digest(text.encode("utf-8")),
                    "text_file": "post-texts/" + post_id + ".txt",
                    "thread_char_start": start, "thread_char_end": end,
                    "body_html_char_start": body.inner_start, "body_html_char_end": body.end,
                    "body_html_sha256": digest(source_html[body.inner_start:body.end].encode("utf-8")),
                    "excluded_body_quotes": quotes, "media": renderer.media, "math_formulas": renderer.math_count}
            (directory / post["text_file"]).write_bytes(text.encode("utf-8"))
            posts.append(post)
            selected.append(post)
            stamp = dates[0] if dates else "时间未载入"
            note = '<p class="meta">正文中的引用原文已省略。</p>' if quotes else ""
            post_html.append(f'<article id="{post_id}"><h2>{html.escape(floor)}</h2>'
                             f'<div class="meta">{html.escape(username)} · {html.escape(stamp)} · '
                             f'<a href="{html.escape(post["source_url"], quote=True)}">来源</a> · {len(text):,} 字符</div>{note}{markup}</article>')
        if not selected or not thread_text.strip():
            raise ValueError(f"No author's text found in {source.name}")
        if len(thread_text) > 200_000:
            raise ValueError("Thread exceeds detector limit; split with an explicit mapping before inference")
        rec = {"schema_version": "author-archive-sample/1.0", "id": sample_id, "title": title,
               "text": thread_text, "source_label": "human", "provenance": "author_declaration",
               "label_evidence": "Author stated in this task that these contents were written manually by them.",
               "author": username, "uid": next(iter(author_ids)), "kind": "author_supplied_forum_archive",
               "source_group": "kechuang-author", "source": url, "archive_file": str(source.resolve()),
               "archive_sha256": digest(raw), "collected_at": now, "snapshot_date": str(message.get("Date")),
               "characters": len(thread_text), "sha256": digest(thread_text.encode("utf-8")),
               "text_file": "texts/" + sample_id + ".txt", "readable_file": "readable/" + sample_id + ".html",
               "post_ids": [p["id"] for p in selected], "posts": len(selected),
               "website_ai_notices": notices, "license": "author_supplied_local_use_only_no_redistribution_grant",
               "training_overlap": "not_audited", "transformations": ["MIME transfer decoding and strict HTML charset decoding",
                   "Select rendered body by exact author header; remove blockquotes and media/UI text",
                   "Decode HTML entities; retain paragraph/line breaks; normalize presentation whitespace and remove invisible editor direction markers",
                   "Recover MathJax LaTeX once; keep emoji; concatenate own posts with two newlines",
                   "No OCR, external fetch, prose rewriting, or detector-based selection"]}
        records.append(rec)
        (directory / rec["text_file"]).write_bytes(thread_text.encode("utf-8"))
        introduction = '<nav><a href="../index.html">返回目录</a></nav>' + f'<h1>{html.escape(title)}</h1>'
        introduction += f'<p class="meta">{len(selected)} 条本人帖子；检测纯文本见 <a href="../{rec["text_file"]}">TXT</a>。图片只用于阅读。</p>'
        if notices:
            introduction += '<p class="meta">来源标签 human 按作者本次声明记录；原网页另有历史提示：' + html.escape("；".join(notices)) + '</p>'
        (directory / rec["readable_file"]).write_text(document(title, introduction + "".join(post_html)), encoding="utf-8")
        evidence.append({"url": url, "archive_file": str(source.resolve()), "sha256": digest(raw), "bytes": len(raw),
                         "html_sha256": digest(payload), "charset": charset, "html_parts": len(candidates),
                         "snapshot_date": str(message.get("Date")), "rendered_posts": len(bodies),
                         "selected_posts": len(selected), "excluded_other_authors": len(bodies) - len(selected),
                         "website_ai_notices": notices, "author_selection_audit": audit,
                         "scope": "rendered posts present in this local MHTML only; unseen pages are not fetched"})
        index_rows.append(f'<tr><td><a href="{rec["readable_file"]}">{html.escape(title)}</a></td>'
                          f'<td>{len(selected)}</td><td>{len(thread_text):,}</td><td><a href="{rec["text_file"]}">TXT</a></td></tr>')
        print(f"{sample_id}: {len(selected)} own posts, {len(thread_text)} characters", flush=True)
    if len(author_ids) != 1:
        raise ValueError("Author name maps to multiple account IDs")
    write_jsonl(directory / "samples.jsonl", records)
    write_jsonl(directory / "posts.jsonl", posts)
    write_json(directory / "sources/archives.json", evidence)
    write_json(directory / "sources/assets.json", list(assets.values()))
    collection = {"schema_version": "author-archive-collection/1.0", "created_at": now,
                  "author": username, "uid": next(iter(author_ids)), "source_label": "human",
                  "label_provenance": "author_declaration_before_detection", "samples": len(records), "posts": len(posts),
                  "text_posts": sum(bool(p["text"]) for p in posts), "characters": sum(r["characters"] for r in records),
                  "source_label_counts": dict(Counter(r["source_label"] for r in records)),
                  "samples_sha256": digest((directory / "samples.jsonl").read_bytes()),
                  "posts_sha256": digest((directory / "posts.jsonl").read_bytes()),
                  "assets": len(assets), "upstream_evidence": evidence,
                  "website_notice_threads": [r["id"] for r in records if r["website_ai_notices"]],
                  "detector_configuration_fixed_before_scoring": {"policy": "thesis-cnki-20260929-v2",
                      "models": "all_ten", "segment_chars": 400, "min_segment_chars": 120,
                      "threshold_overrides": False, "source_labels_used_as_model_inputs": False},
                  "extraction_script_sha256": digest(Path(__file__).read_bytes())}
    write_json(directory / "collection.json", collection)
    body = f'<h1>{html.escape(username)}的科创帖子</h1><p>{len(records)} 个主题，{len(posts)} 条本人帖子，{collection["characters"]:,} 个检测输入字符。</p>'
    body += '<p>来源标签：human（作者本次明确声明）。只提取存档里该账号的正文。引用、网页控件和附件文字不计入检测；图片保留用于离线阅读。</p>'
    body += '<table><thead><tr><th>主题</th><th>本人帖子</th><th>输入字符</th><th>纯文本</th></tr></thead><tbody>' + "".join(index_rows) + '</tbody></table>'
    (directory / "index.html").write_text(document(username + "的科创帖子", body), encoding="utf-8")
    write_json(directory / "COMPLETE.json", {"status": "complete", "samples": len(records), "posts": len(posts),
               "samples_sha256": collection["samples_sha256"], "posts_sha256": collection["posts_sha256"]})
    verify(directory)


def verify(directory):
    sys.path.insert(0, str(ROOT / "scripts"))
    from public_samples import verify as verify_samples
    records, collection = verify_samples(directory)
    posts = [json.loads(line) for line in (directory / "posts.jsonl").read_text(encoding="utf-8").splitlines()]
    if digest((directory / "posts.jsonl").read_bytes()) != collection["posts_sha256"]:
        raise ValueError("Post JSONL hash mismatch")
    if len(posts) != collection["posts"] or len({p["id"] for p in posts}) != len(posts):
        raise ValueError("Post count or IDs invalid")
    complete = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete["status"] != "complete" or complete["samples_sha256"] != collection["samples_sha256"]:
        raise ValueError("Collection completion marker mismatch")
    for rec in records:
        members = [p for p in posts if p["thread_sample_id"] == rec["id"]]
        if [p["id"] for p in members] != rec["post_ids"]:
            raise ValueError("Thread membership differs from metadata")
        if "\n\n".join(p["text"] for p in members if p["text"]) != rec["text"]:
            raise ValueError("Thread is not exactly the selected post text")
        for post in members:
            if post["uid"] != collection["uid"] or post["author"] != collection["author"]:
                raise ValueError("Author identity mismatch")
            raw = (directory / post["text_file"]).read_bytes()
            if raw != post["text"].encode("utf-8") or digest(raw) != post["sha256"]:
                raise ValueError("Post text/hash mismatch")
            if rec["text"][post["thread_char_start"]:post["thread_char_end"]] != post["text"]:
                raise ValueError("Post-to-thread offset mismatch")
    for source in collection["upstream_evidence"]:
        if digest(Path(source["archive_file"]).read_bytes()) != source["sha256"]:
            raise ValueError("Original MHTML archive changed")
    for asset in json.loads((directory / "sources/assets.json").read_text(encoding="utf-8")):
        if digest((directory / asset["path"]).read_bytes()) != asset["sha256"]:
            raise ValueError("Extracted image hash mismatch")
    print(f"Verified {len(posts)} author posts, original archive hashes, position maps and images", flush=True)
    return records, collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "verify"))
    parser.add_argument("--source-dir", type=Path, default=ROOT / "temp/20260930 kechuang")
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "samples/kechuang-human-20260930")
    parser.add_argument("--username", default="三水合番")
    args = parser.parse_args()
    if args.action == "collect":
        collect(args.source_dir.resolve(), args.sample_dir.resolve(), args.username)
    else:
        verify(args.sample_dir.resolve())


if __name__ == "__main__":
    main()
