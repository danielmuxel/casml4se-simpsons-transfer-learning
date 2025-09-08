#!/usr/bin/env python3
"""
Generate an HTML overview page for AI inference outputs.

- Scans a target subfolder under data/inference/output/<subfolder>
- Expects pairs of image files and JSON files (.json) sharing the same basename
- Produces an index.html in that subfolder, using Tailwind via CDN

Usage:
  python tools/generate_overview.py --subfolder linear_probe

Optional:
  --title "My Title"
  --open  # opens the generated file in default browser
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "inference" / "output"


@dataclasses.dataclass
class InferenceItem:
    image_path: Path
    json_path: Path
    prediction_index: Optional[int]
    prediction_label: Optional[str]
    prediction_probability: Optional[float]
    expected_label: Optional[str] = None
    correctness: Optional[str] = None  # 'correct' | 'incorrect' | 'unknown'
    file_basename: Optional[str] = None


def load_classes() -> List[str]:
    """Load class list from data/processed/classes.json if present."""
    classes_path = ROOT / "data" / "processed" / "classes.json"
    if not classes_path.exists():
        return []
    try:
        data = json.loads(classes_path.read_text())
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return []


def infer_expected_label(image_path: Optional[Path], classes: List[str]) -> Optional[str]:
    if not image_path or not classes:
        return None
    stem = image_path.stem
    if not stem:
        return None
    candidates: List[Tuple[int, str]] = []
    for cls in classes:
        if not cls:
            continue
        # Prefer prefix match like "<class>_..."
        if stem.startswith(cls + "_") or stem == cls:
            candidates.append((2, cls))
        # Exact token presence (underscore delimited)
        token = f"_{cls}_"
        if token in f"_{stem}_":
            candidates.append((1, cls))
        # Substring fallback
        if cls in stem:
            candidates.append((0, cls))
    if not candidates:
        return None
    # Rank by rule weight then by length (longest label wins)
    candidates.sort(key=lambda t: (t[0], len(t[1])), reverse=True)
    return candidates[0][1]


def find_items(subfolder: str) -> List[InferenceItem]:
    target_dir = OUTPUT_ROOT / subfolder
    if not target_dir.exists() or not target_dir.is_dir():
        raise SystemExit(f"Target directory not found: {target_dir}")

    items: List[InferenceItem] = []
    # Map basenames to potential image/json
    images: Dict[str, Path] = {}
    jsons: Dict[str, Path] = {}

    for entry in target_dir.iterdir():
        if entry.is_file():
            if entry.suffix.lower() == ".json":
                jsons[entry.stem] = entry
            elif entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                images[entry.stem] = entry

    # Load classes for expected label inference
    classes = load_classes()

    # Pair by basename
    for stem, json_path in jsons.items():
        image_path = images.get(stem)
        # Allow JSONs without images; still list them
        prediction_index: Optional[int] = None
        prediction_label: Optional[str] = None
        prediction_probability: Optional[float] = None
        expected_label: Optional[str] = None
        correctness: Optional[str] = None

        try:
            data = json.loads(json_path.read_text())
            prediction_index = int(data.get("prediction_index")) if data.get("prediction_index") is not None else None
            prediction_label = data.get("prediction_label")
            prob = data.get("prediction_probability")
            prediction_probability = float(prob) if prob is not None else None
            # If image_path is in JSON, prefer that relative path if file exists
            json_image_path = data.get("image_path")
            if json_image_path:
                candidate = ROOT / json_image_path
                if candidate.exists():
                    image_path = candidate
        except Exception:
            # Keep None fields if JSON malformed
            pass

        if image_path is None:
            # try to infer with .jpg if missing
            guessed = target_dir / f"{stem}.jpg"
            if guessed.exists():
                image_path = guessed
        # Derive file basename for display purposes
        file_basename = (image_path.name if image_path and image_path.name else json_path.name)
        # Infer expected label from filename, if possible
        expected_label = infer_expected_label(image_path, classes)
        if expected_label and prediction_label:
            correctness = "correct" if prediction_label == expected_label else "incorrect"
        else:
            correctness = "unknown"

        items.append(
            InferenceItem(
                image_path=image_path if image_path else Path(stem),
                json_path=json_path,
                prediction_index=prediction_index,
                prediction_label=prediction_label,
                prediction_probability=prediction_probability,
                expected_label=expected_label,
                correctness=correctness,
                file_basename=file_basename,
            )
        )

    # Sort by filename for stability
    items.sort(key=lambda it: it.json_path.name)
    return items


def render_html(subfolder: str, items: List[InferenceItem], title: Optional[str]) -> str:
    page_title = title or f"Inference Overview — {subfolder}"
    safe_title = html.escape(page_title)

    # Build cards
    cards: List[str] = []
    for it in items:
        label = html.escape(it.prediction_label) if it.prediction_label is not None else "—"
        prob = f"{it.prediction_probability:.4f}" if it.prediction_probability is not None else "—"
        idx = str(it.prediction_index) if it.prediction_index is not None else "—"
        file_name = html.escape(it.file_basename) if it.file_basename else html.escape(it.json_path.name)

        # Style the predicted label based on correctness
        label_classes = "font-medium"
        if it.correctness == "correct":
            label_classes = "inline-flex items-center rounded bg-green-600 px-2 py-0.5 text-xs font-semibold text-white"
        elif it.correctness == "incorrect":
            label_classes = "inline-flex items-center rounded bg-red-600 px-2 py-0.5 text-xs font-semibold text-white"

        # Style the border based on correctness
        border_class = (
            "border-green-600" if it.correctness == "correct"
            else ("border-red-600" if it.correctness == "incorrect" else "border-gray-200")
        )
        article_classes = f"group border-4 {border_class} rounded-lg shadow-sm overflow-hidden bg-white focus-within:ring-2 focus-within:ring-blue-500"

        img_src: str
        # Prefer relative path from the output folder for portability
        try:
            img_src = os.path.relpath(it.image_path, (OUTPUT_ROOT / subfolder)) if it.image_path else ""
        except Exception:
            img_src = str(it.image_path)

        if not img_src:
            # Fallback to JSON path basename (no image)
            img_alt = html.escape(it.json_path.name)
            img_tag = f"<div class=\"w-full h-48 bg-gray-100 flex items-center justify-center text-gray-500 text-sm\" aria-label=\"No image available for {img_alt}\">No image</div>"
        else:
            img_alt = html.escape(it.json_path.name)
            img_tag = (
                f"<img src=\"{html.escape(img_src)}\" alt=\"{img_alt}\" "
                f"class=\"w-full h-48 object-contain bg-gray-50\" loading=\"lazy\">"
            )

        json_rel = os.path.relpath(it.json_path, (OUTPUT_ROOT / subfolder))
        json_link = f"<a href=\"{html.escape(json_rel)}\" class=\"text-blue-600 hover:underline\" target=\"_blank\" rel=\"noopener noreferrer\">JSON</a>"

        cards.append(
            """
            <article class="{article_classes}" tabindex="0" role="article" aria-label="Inference item">
              <div class="bg-white">
                {img_tag}
              </div>
              <div class="p-4 space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Index</span>
                  <span class="font-mono text-sm">{idx}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Label</span>
                  <span class="{label_classes}">{label}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Probability</span>
                  <span class="font-mono">{prob}</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">File</span>
                  <span class="font-mono text-xs text-gray-600">{file_name}</span>
                </div>
                <div class="pt-2">
                  {json_link}
                </div>
              </div>
            </article>
            """.replace("{img_tag}", img_tag).format(idx=idx, label=label, prob=prob, json_link=json_link, label_classes=label_classes, file_name=file_name, article_classes=article_classes)
        )

    cards_html = "\n".join(cards) if cards else (
        """
        <div class="text-gray-600">No items found. Ensure JSON files are present.</div>
        """
    )

    # Simple JS to allow refresh without re-running the script if new files are added
    # This fetches directory listing via generated manifest.json for dynamic updates
    script = """
    <script>
    // If a manifest.json is present, provide a refresh button to update the grid
    async function handleRefresh() {
      const res = await fetch('manifest.json', { cache: 'no-cache' });
      if (!res.ok) {
        alert('No manifest.json found. Re-run the generator to refresh.');
        return;
      }
      const data = await res.json();
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      for (const item of data.items) {
        const article = document.createElement('article');
        const borderClass = item.correctness === 'correct' ? 'border-green-600' : (item.correctness === 'incorrect' ? 'border-red-600' : 'border-gray-200');
        article.className = `group border-4 ${borderClass} rounded-lg shadow-sm overflow-hidden bg-white focus-within:ring-2 focus-within:ring-blue-500`;
        article.tabIndex = 0;
        article.setAttribute('role', 'article');
        article.setAttribute('aria-label', 'Inference item');

        const imgWrap = document.createElement('div');
        imgWrap.className = 'bg-white';
        if (item.image_rel) {
          const img = document.createElement('img');
          img.src = item.image_rel;
          img.alt = item.json_name;
          img.loading = 'lazy';
          img.className = 'w-full h-48 object-contain bg-gray-50';
          imgWrap.appendChild(img);
        } else {
          const noimg = document.createElement('div');
          noimg.className = 'w-full h-48 bg-gray-100 flex items-center justify-center text-gray-500 text-sm';
          noimg.setAttribute('aria-label', `No image available for ${item.json_name}`);
          noimg.textContent = 'No image';
          imgWrap.appendChild(noimg);
        }

        const body = document.createElement('div');
        body.className = 'p-4 space-y-2';

        const row = (label, value, mono=false, extraClass='') => {
          const r = document.createElement('div');
          r.className = 'flex items-center justify-between';
          const l = document.createElement('span');
          l.className = 'text-sm text-gray-500';
          l.textContent = label;
          const v = document.createElement('span');
          v.className = mono ? 'font-mono text-sm' : (label==='Label' ? 'font-medium' : '');
          if (extraClass) v.className += ` ${extraClass}`;
          v.textContent = value;
          r.append(l, v);
          return r;
        };

        const labelClass = item.correctness === 'correct'
          ? 'inline-flex items-center rounded bg-green-600 px-2 py-0.5 text-xs font-semibold text-white'
          : (item.correctness === 'incorrect'
            ? 'inline-flex items-center rounded bg-red-600 px-2 py-0.5 text-xs font-semibold text-white'
            : '');

        body.append(
          row('Index', item.prediction_index ?? '—', true),
          row('Label', item.prediction_label ?? '—', false, labelClass),
          row('Probability', item.prediction_probability_str ?? '—', true),
          row('File', item.file_basename ?? (item.json_name ?? '—'), true, 'text-gray-600 text-xs'),
        );

        const linkWrap = document.createElement('div');
        linkWrap.className = 'pt-2';
        const a = document.createElement('a');
        a.href = item.json_rel;
        a.className = 'text-blue-600 hover:underline';
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = 'JSON';
        linkWrap.appendChild(a);

        body.appendChild(linkWrap);

        article.append(imgWrap, body);
        grid.appendChild(article);
      }
    }
    </script>
    """

    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{safe_title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body class="bg-gray-50">
    <main class="max-w-7xl mx-auto p-6">
      <header class="flex items-center justify-between mb-6">
        <h1 class="text-2xl font-semibold">{safe_title}</h1>
        <div class="space-x-2">
          <button onclick="location.reload()" class="px-3 py-1.5 rounded bg-gray-200 hover:bg-gray-300 text-gray-800" aria-label="Reload the page" title="Reload the page">Reload</button>
          <button onclick="handleRefresh()" class="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-700 text-white" aria-label="Refresh from manifest" title="Refresh from manifest.json">Refresh</button>
        </div>
      </header>
      <section id="grid" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {cards_html}
      </section>
    </main>
    {script}
  </body>
</html>
"""
    return html_doc


def write_manifest(subfolder: str, items: List[InferenceItem]) -> None:
    target_dir = OUTPUT_ROOT / subfolder
    manifest_items: List[Dict[str, Optional[str]]] = []
    for it in items:
        try:
            image_rel = os.path.relpath(it.image_path, target_dir) if it.image_path and it.image_path.exists() else None
        except Exception:
            image_rel = str(it.image_path) if it.image_path else None

        manifest_items.append(
            {
                "json_name": it.json_path.name,
                "json_rel": os.path.relpath(it.json_path, target_dir),
                "image_rel": image_rel,
                "prediction_index": it.prediction_index,
                "prediction_label": it.prediction_label,
                "prediction_probability": it.prediction_probability,
                "prediction_probability_str": f"{it.prediction_probability:.4f}" if it.prediction_probability is not None else None,
                "expected_label": it.expected_label,
                "correctness": it.correctness,
                "file_basename": it.file_basename,
            }
        )

    manifest = {"items": manifest_items}
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate HTML overview for inference outputs")
    parser.add_argument("--subfolder", required=True, help="Subfolder under data/inference/output, e.g. linear_probe")
    parser.add_argument("--title", default=None, help="Custom page title")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the browser")
    args = parser.parse_args(argv)

    items = find_items(args.subfolder)
    html_text = render_html(args.subfolder, items, args.title)

    target_dir = OUTPUT_ROOT / args.subfolder
    out_file = target_dir / "index.html"
    out_file.write_text(html_text)

    write_manifest(args.subfolder, items)

    print(f"Generated: {out_file}")
    if args.open:
        try:
            webbrowser.open(out_file.as_uri())
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


