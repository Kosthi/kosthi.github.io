import path from "node:path";

const blogRoot = path.resolve("src/content/blog");

function walk(node, visit) {
	visit(node);
	if (Array.isArray(node.children)) {
		for (const child of node.children) {
			walk(child, visit);
		}
	}
}

/**
 * Rewrites bundle-relative image URLs to their `public/` location.
 *
 *   src/content/blog/zh/cs336-assign5/index.md   ->  /blog/cs336-assign5/<img>
 *   src/content/blog/en/cs336-assign5/index.md   ->  /blog/cs336-assign5/<img>
 *
 * The language segment is stripped so both translations point at one copy of
 * the image in `public/blog/<slug>/`.
 */
export default function remarkPublicImages() {
	return (tree, file) => {
		if (!file.path) return;

		const relativePath = path.relative(blogRoot, file.path).split(path.sep).join("/");
		const withoutExtension = relativePath.replace(/\.(?:md|mdx)$/, "");
		const withoutIndex = withoutExtension.endsWith("/index")
			? withoutExtension.slice(0, -6)
			: withoutExtension;
		const slug = withoutIndex.replace(/^(?:zh|en)\//, "");

		walk(tree, (node) => {
			if (node.type !== "image" || typeof node.url !== "string") return;
			// Leave absolute URLs, root-relative paths, and fragments alone.
			if (/^(?:[a-z]+:|\/|#)/i.test(node.url)) return;
			node.url = `/blog/${slug}/${node.url.replace(/^\.\//, "")}`;
		});
	};
}
