import { defineConfig } from "astro/config";
import { unified, rehypeHeadingIds } from "@astrojs/markdown-remark";
import sitemap from "@astrojs/sitemap";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkPublicImages from "./src/plugins/remark-public-images.mjs";

// https://astro.build/config
export default defineConfig({
  site: "https://koschei.top",
  integrations: [
    sitemap({
      i18n: {
        defaultLocale: "zh",
        locales: { zh: "zh-CN", en: "en" },
      },
    }),
  ],
  output: "static",
  trailingSlash: "always",
  compressHTML: true,
  markdown: {
    processor: unified({
      remarkPlugins: [remarkPublicImages, remarkMath],
      rehypePlugins: [rehypeHeadingIds, rehypeKatex],
    }),
    shikiConfig: {
      themes: {
        light: "github-light",
        dark: "github-dark",
      },
    },
  },
});
