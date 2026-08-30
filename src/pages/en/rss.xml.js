import rss from "@astrojs/rss";
import { SITE_TITLE } from "../../consts";
import { UI, getPosts, localizePath, slugOf } from "../../i18n";

const lang = "en";

export async function GET(context) {
	const posts = await getPosts(lang);
	return rss({
		title: SITE_TITLE,
		description: UI[lang].siteDescription,
		site: context.site,
		customData: "<language>en</language>",
		items: posts.map((post) => ({
			title: post.data.title,
			description: post.data.description,
			pubDate: post.data.pubDate,
			categories: [...post.data.tags, ...post.data.categories],
			link: localizePath(lang, `/blog/${slugOf(post)}/`),
		})),
	});
}
