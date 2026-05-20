// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import mermaid from "astro-mermaid";

// https://astro.build/config
export default defineConfig({
  site: "https://fujishigetemma.github.io",
  base: "/edmkit-search",
  integrations: [
    mermaid({
      theme: "neutral",
      autoTheme: true,
    }),
    starlight({
      title: "edmkit-search",
      description: "Trajectory construction on an energy landscape, packaged as a library on top of edmkit",
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/FujishigeTemma/edmkit-search" },
      ],
      sidebar: [
        { label: "Getting Started", slug: "getting-started" },
        {
          label: "Concepts",
          items: [
            { label: "The Search Loop", slug: "concepts/search-loop" },
            { label: "Dataset", slug: "concepts/dataset" },
            { label: "Energy", slug: "concepts/energy" },
            { label: "Neighborhood", slug: "concepts/neighborhood" },
            { label: "Strategy", slug: "concepts/strategy" },
          ],
        },
        {
          label: "API Reference",
          autogenerate: { directory: "reference" },
        },
      ],
      editLink: {
        baseUrl: "https://github.com/FujishigeTemma/edmkit-search/edit/main/website/",
      },
      lastUpdated: true,
    }),
  ],
});
