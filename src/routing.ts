import { getCollection, type CollectionEntry } from "astro:content";
import { LANGS, getPosts, langOf, normalizeTerm, slugOf, type Lang } from "./i18n";

/** All published posts across every language, used to reason about translations. */
async function allPosts(): Promise<CollectionEntry<"blog">[]> {
	const perLang = await Promise.all(LANGS.map((lang) => getPosts(lang)));
	return perLang.flat();
}

/** slug -> the languages it has been written in. */
export async function translationMap(): Promise<Map<string, Lang[]>> {
	const map = new Map<string, Lang[]>();
	for (const post of await allPosts()) {
		const slug = slugOf(post);
		map.set(slug, [...(map.get(slug) ?? []), langOf(post)]);
	}
	return map;
}

/** Static paths for `/blog/[slug]/` in one language. */
export async function blogPaths(lang: Lang) {
	const translations = await translationMap();
	const posts = await getPosts(lang);
	return posts.map((post) => {
		const slug = slugOf(post);
		return {
			params: { slug },
			props: { post, availableLangs: translations.get(slug) ?? [lang] },
		};
	});
}

type TermField = "tags" | "categories";

/**
 * Terms used by a language's posts, keyed by their normalized form so that
 * `LLM` and `llm` collapse into one page while keeping the authored casing.
 */
export async function collectTerms(lang: Lang, field: TermField) {
	const terms = new Map<string, { name: string; count: number }>();
	for (const post of await getPosts(lang)) {
		for (const value of post.data[field]) {
			const key = normalizeTerm(value);
			const existing = terms.get(key);
			if (existing) existing.count += 1;
			else terms.set(key, { name: value, count: 1 });
		}
	}
	return [...terms.values()].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

/** Static paths for `/tags/[tag]/` and `/categories/[category]/`. */
export async function termPaths(lang: Lang, field: TermField, param: "tag" | "category") {
	// A term only gets a page in a language where posts actually use it, so the
	// language switcher never links to a 404.
	const perLang = await Promise.all(
		LANGS.map(async (code) => [code, await collectTerms(code, field)] as const),
	);
	const availability = new Map<string, Lang[]>();
	for (const [code, terms] of perLang) {
		for (const term of terms) {
			const key = normalizeTerm(term.name);
			availability.set(key, [...(availability.get(key) ?? []), code]);
		}
	}

	const terms = await collectTerms(lang, field);
	return Promise.all(
		terms.map(async (term) => {
			const key = normalizeTerm(term.name);
			const posts = (await getPosts(lang)).filter((post) =>
				post.data[field].some((value) => normalizeTerm(value) === key),
			);
			return {
				params: { [param]: key },
				props: { name: term.name, posts, availableLangs: availability.get(key) ?? [lang] },
			};
		}),
	);
}

/** Load a standalone page (`home`, `about`) for a language. */
export async function getPage(lang: Lang, name: string) {
	const entries = await getCollection("pages", ({ id }) => id === `${lang}/${name}`);
	const entry = entries[0];
	if (!entry) throw new Error(`missing page content: src/content/pages/${lang}/${name}.md`);
	return entry;
}
