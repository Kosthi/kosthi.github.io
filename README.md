# kosthi.github.io

[koschei.top](https://koschei.top) 的源码。中英双语，用 [Astro](https://astro.build) 构建，部署在 GitHub Pages。

## 开发

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # 构建到 dist/
npm run preview  # 预览构建产物
npm run check    # 类型检查
```

## 目录结构

```
src/
├── content/
│   ├── blog/<lang>/<slug>/index.md   文章，lang 为 zh 或 en
│   └── pages/<lang>/{home,about}.md  首页简介与关于页正文
├── components/                        UI 组件
├── layouts/                           页面骨架
├── views/                             按语言复用的页面主体
├── pages/                             路由：中文在根，英文在 /en/
├── i18n.ts                            语言配置、界面文案、内容查询
├── routing.ts                         静态路径与标签/分类聚合
└── styles/global.css                  设计变量与排版
public/
├── blog/<slug>/                       文章配图（两种语言共用一份）
└── CNAME                              自定义域名
```

## 写文章

在 `src/content/blog/zh/<slug>/index.md` 新建文件：

```markdown
---
title: "标题"
description: "一句话摘要，会显示在列表卡片和 SEO 里"
pubDate: 2026-01-01T00:00:00+08:00
tags: ["标签"]
categories: ["分类"]
---
```

可选字段：`updatedDate`、`heroImage`、`socialImage`、`enable_katex`（需要 LaTeX 公式时设为 `true`）、`draft`（`true` 时仅在 `npm run dev` 可见）。

英文版放在 `src/content/blog/en/<slug>/index.md`，**slug 相同**即自动互链并生成 `hreflang`。只写一种语言也可以，语言切换按钮会自动隐藏。

图片放 `public/blog/<slug>/`，正文里直接写 `![说明](image.png)`，构建时会自动补全为 `/blog/<slug>/image.png`。

## URL 结构

| 路径 | 说明 |
|---|---|
| `/`、`/en/` | 首页 |
| `/blog/`、`/en/blog/` | 文章列表 |
| `/blog/<slug>/` | 文章 |
| `/tags/`、`/categories/` | 标签与分类 |
| `/rss.xml`、`/en/rss.xml` | 订阅源 |

## 部署

推送到 `master` 由 `.github/workflows/deploy.yml` 自动构建部署。仓库 **Settings → Pages → Source 需设为 "GitHub Actions"**。

## 致谢

视觉设计改编自 [skyzh-site](https://github.com/skyzh/skyzh-site) 和 [Astro Nano](https://github.com/markhorn-dev/astro-nano)，均为 MIT 许可，详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

上游可追踪：

```bash
git remote add skyzh https://github.com/skyzh/skyzh-site.git
git fetch skyzh
git diff skyzh/main -- src/styles/global.css
```
