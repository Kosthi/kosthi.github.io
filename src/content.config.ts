import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

/**
 * Posts are stored as `src/content/blog/<lang>/<slug>/index.md`, so the entry
 * id doubles as the language tag: `zh/cs336-assign1`, `en/cs336-assign1`.
 */
const blog = defineCollection({
	loader: glob({
		pattern: ["**/*.md", "**/*.mdx"],
		base: "./src/content/blog",
		generateId: ({ entry }) => entry.replace(/(?:\/index)?\.(?:md|mdx)$/, ""),
	}),
	schema: z.object({
		title: z.string(),
		description: z.string().default(""),
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		heroImage: z.string().optional(),
		socialImage: z.string().optional(),
		tags: z.array(z.string()).default([]),
		categories: z.array(z.string()).default([]),
		enable_katex: z.boolean().optional(),
		draft: z.boolean().default(false),
	}),
});

/**
 * Standalone pages (the homepage intro, the about page) live here so the prose
 * is editable as Markdown instead of being buried in components.
 * Ids follow the same `<lang>/<name>` convention as posts.
 */
const pages = defineCollection({
	loader: glob({
		pattern: ["**/*.md", "**/*.mdx"],
		base: "./src/content/pages",
		generateId: ({ entry }) => entry.replace(/\.(?:md|mdx)$/, ""),
	}),
	schema: z.object({
		title: z.string(),
		description: z.string().default(""),
	}),
});

export const collections = { blog, pages };
