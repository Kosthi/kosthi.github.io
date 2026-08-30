import { getCollection, type CollectionEntry } from "astro:content";

export const DEFAULT_LANG = "zh";
export const LANGS = ["zh", "en"] as const;

export type Lang = (typeof LANGS)[number];

/**
 * The default locale lives at the site root and English is mirrored under
 * `/en/`, which matches the URL layout the Hugo site used.
 */
export const LOCALES: Record<Lang, { htmlLang: string; label: string; dateLocale: string }> = {
	zh: { htmlLang: "zh-CN", label: "中文", dateLocale: "zh-CN" },
	en: { htmlLang: "en", label: "English", dateLocale: "en-US" },
};

export const UI = {
	zh: {
		siteDescription: "Koschei 的博客，记录数据库、系统与大模型方向的学习与实践。",
		nav: { blog: "文章", tags: "标签", categories: "分类", about: "关于" },
		latestPosts: "最新文章",
		seeAllPosts: "全部文章",
		allPosts: "文章",
		allTags: "标签",
		allCategories: "分类",
		backToBlog: "← 返回文章列表",
		lastUpdated: "最后更新于",
		tags: "标签",
		categories: "分类",
		comments: "评论",
		backToTop: "回到顶部",
		postsIn: (name: string) => `包含「${name}」的文章`,
		postCount: (n: number) => `${n} 篇`,
		noTranslation: "本文暂无英文版本。",
		readMore: "阅读全文",
	},
	en: {
		siteDescription: "Koschei's blog on databases, systems, and large language models.",
		nav: { blog: "blog", tags: "tags", categories: "categories", about: "about" },
		latestPosts: "Latest posts",
		seeAllPosts: "See all posts",
		allPosts: "Blog",
		allTags: "Tags",
		allCategories: "Categories",
		backToBlog: "← Back to blog",
		lastUpdated: "Last updated on",
		tags: "Tags",
		categories: "Categories",
		comments: "Comments",
		backToTop: "Back to top",
		postsIn: (name: string) => `Posts in ${name}`,
		postCount: (n: number) => `${n} post${n === 1 ? "" : "s"}`,
		noTranslation: "This post is not available in Chinese.",
		readMore: "Read more",
	},
} as const;

/** Prefix a root-relative path with the language segment. */
export function localizePath(lang: Lang, pathname: string): string {
	const normalised = pathname.startsWith("/") ? pathname : `/${pathname}`;
	return lang === DEFAULT_LANG ? normalised : `/${lang}${normalised}`;
}

/** Reverse of `localizePath`: strip the language segment back off a URL. */
export function delocalizePath(pathname: string): string {
	for (const lang of LANGS) {
		if (lang === DEFAULT_LANG) continue;
		if (pathname === `/${lang}` || pathname === `/${lang}/`) return "/";
		if (pathname.startsWith(`/${lang}/`)) return pathname.slice(lang.length + 1);
	}
	return pathname;
}

export function getLangFromUrl(url: URL): Lang {
	const segment = url.pathname.split("/").filter(Boolean)[0];
	return LANGS.includes(segment as Lang) ? (segment as Lang) : DEFAULT_LANG;
}

/** Entry ids are `<lang>/<slug>`; the slug is what appears in the URL. */
export function slugOf(entry: CollectionEntry<"blog">): string {
	return entry.id.replace(/^(?:zh|en)\//, "");
}

export function langOf(entry: CollectionEntry<"blog">): Lang {
	return entry.id.startsWith("en/") ? "en" : "zh";
}

/** Published posts for one language, newest first. Drafts are excluded. */
export async function getPosts(lang: Lang): Promise<CollectionEntry<"blog">[]> {
	const posts = await getCollection("blog", ({ id, data }) => {
		if (import.meta.env.PROD && data.draft) return false;
		return id.startsWith(`${lang}/`);
	});
	return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

/** Slugs that exist in a given language, used to decide if a post is translated. */
export async function getTranslatedSlugs(lang: Lang): Promise<Set<string>> {
	return new Set((await getPosts(lang)).map(slugOf));
}

/**
 * Tags/categories are matched case-insensitively and slugified for URLs, while
 * the authored casing is what gets displayed. Multi-word English tags such as
 * "performance optimization" would otherwise produce a URL containing a space.
 */
export function normalizeTerm(term: string): string {
	return term
		.trim()
		.toLowerCase()
		.replace(/\s+/g, "-")
		.replace(/[/?#[\]@!$&'()*+,;=]/g, "");
}
