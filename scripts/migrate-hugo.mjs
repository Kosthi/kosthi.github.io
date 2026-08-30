// One-off migration: Hugo (LoveIt) page bundles -> Astro content collections.
//
//   content/posts/<Slug>/index.zh-cn.md  ->  src/content/blog/zh/<slug>/index.md
//   content/posts/<Slug>/index.en.md     ->  src/content/blog/en/<slug>/index.md
//   content/posts/<Slug>/*.png           ->  public/blog/<slug>/*.png
//
// Handles both TOML (+++) and YAML (---) front matter, since the posts use a
// mix of the two. Delete this script once the migration is reviewed.

import fs from "node:fs";
import path from "node:path";

const POSTS_DIR = "content/posts";
const OUT_DIR = "src/content/blog";
const IMAGE_DIR = "public/blog";

const LANGS = { "zh-cn": "zh", en: "en" };

/** Strip surrounding quotes from a TOML/YAML scalar. */
function unquote(value) {
	const trimmed = value.trim();
	const match = /^(['"])([\s\S]*)\1$/.exec(trimmed);
	return match ? match[2] : trimmed;
}

/** Parse a `["a", "b"]` inline array. */
function parseArray(value) {
	const inner = value.trim().replace(/^\[/, "").replace(/\]$/, "").trim();
	if (!inner) return [];
	return inner
		.split(",")
		.map((item) => unquote(item))
		.filter(Boolean);
}

/**
 * Both front matter dialects here are flat `key = value` / `key: value` pairs,
 * so one line-based parser covers them. Nested tables (`resources`) are dropped
 * on purpose -- nothing in the content uses them.
 *
 * Some bundles are CRLF, and `.` never matches `\r` in JS, so line endings are
 * normalised before anything else touches the text.
 */
function parseFrontMatter(input) {
	const raw = input.replace(/\r\n/g, "\n");
	const fence = raw.startsWith("+++") ? "+++" : "---";
	if (!raw.startsWith(fence)) throw new Error("missing front matter");
	const end = raw.indexOf(`\n${fence}`, fence.length);
	if (end === -1) throw new Error("unterminated front matter");

	const block = raw.slice(fence.length, end);
	const body = raw.slice(end + fence.length + 1).replace(/^\n+/, "");

	const data = {};
	for (const line of block.split("\n")) {
		// Skip comments, blank lines, and any indented (nested) keys.
		if (!line.trim() || /^\s*#/.test(line) || /^\s/.test(line)) continue;
		const match = /^([A-Za-z_][\w-]*)\s*[:=]\s*(.*)$/.exec(line);
		if (!match) continue;
		const [, key, rest] = match;
		if (!rest) continue;
		data[key] = rest.trim().startsWith("[") ? parseArray(rest) : unquote(rest);
	}
	return { data, body };
}

/** Fall back to the first real paragraph when a post has no description. */
function deriveDescription(body) {
	for (const block of body.split(/\n\s*\n/)) {
		const text = block.trim();
		if (!text || text.startsWith("#") || text.startsWith("```")) continue;
		if (text.startsWith("![") || text.startsWith(">") || text.startsWith("|")) continue;
		const plain = text
			.replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1") // links / images -> label
			.replace(/[*_`>#]/g, "")
			.replace(/\s+/g, " ")
			.trim();
		if (plain.length < 12) continue;
		return plain.length > 110 ? `${plain.slice(0, 110).trimEnd()}…` : plain;
	}
	return "";
}

/** YAML-quote a string for the generated front matter. */
function yamlString(value) {
	return `"${String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

function toIsoDate(value) {
	const date = new Date(value);
	if (Number.isNaN(date.valueOf())) throw new Error(`bad date: ${value}`);
	return date.toISOString();
}

const summary = [];

for (const dirent of fs.readdirSync(POSTS_DIR, { withFileTypes: true })) {
	if (!dirent.isDirectory()) continue;
	const bundle = path.join(POSTS_DIR, dirent.name);
	// Hugo lowercases the URL segment, so the slug follows suit.
	const slug = dirent.name.toLowerCase();

	// Images are shared across translations: copy once into public/blog/<slug>/.
	const images = fs
		.readdirSync(bundle)
		.filter((name) => !name.endsWith(".md"));
	if (images.length > 0) {
		fs.mkdirSync(path.join(IMAGE_DIR, slug), { recursive: true });
		for (const image of images) {
			fs.copyFileSync(path.join(bundle, image), path.join(IMAGE_DIR, slug, image));
		}
	}

	for (const [hugoLang, lang] of Object.entries(LANGS)) {
		const source = path.join(bundle, `index.${hugoLang}.md`);
		if (!fs.existsSync(source)) continue;

		const { data, body } = parseFrontMatter(fs.readFileSync(source, "utf8"));
		const description = data.description || data.subtitle || deriveDescription(body);
		const hasDisplayMath = /\$\$/.test(body);

		const lines = [
			"---",
			`title: ${yamlString(data.title ?? slug)}`,
			`description: ${yamlString(description)}`,
			`pubDate: ${toIsoDate(data.date)}`,
		];
		// Only keep lastmod when it actually differs from the publish date.
		if (data.lastmod && toIsoDate(data.lastmod) !== toIsoDate(data.date)) {
			lines.push(`updatedDate: ${toIsoDate(data.lastmod)}`);
		}
		if (data.tags?.length) {
			lines.push(`tags: [${data.tags.map(yamlString).join(", ")}]`);
		}
		if (data.categories?.length) {
			lines.push(`categories: [${data.categories.map(yamlString).join(", ")}]`);
		}
		if (hasDisplayMath) lines.push("enable_katex: true");
		if (data.draft === "true" || data.draft === true) lines.push("draft: true");
		lines.push("---", "");

		const target = path.join(OUT_DIR, lang, slug, "index.md");
		fs.mkdirSync(path.dirname(target), { recursive: true });
		// Normalise `./image.png` -> `image.png`; remark-public-images prefixes the rest.
		const normalisedBody = body.replace(/(!\[[^\]]*\]\()\.\//g, "$1");
		fs.writeFileSync(target, `${lines.join("\n")}${normalisedBody.trimEnd()}\n`);

		summary.push({ lang, slug, katex: hasDisplayMath, derived: !data.description });
	}
}

console.table(summary);
console.log(`migrated ${summary.length} files across ${new Set(summary.map((s) => s.slug)).size} posts`);
