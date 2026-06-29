// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax/svg';

// Project (not user) GitHub Pages site -> served from a sub-path.
// site + base together give correct canonical URLs and asset paths for SEO.
export default defineConfig({
  site: 'https://isnoobgrammer.github.io',
  base: '/triton-kernel-fused',
  trailingSlash: 'always',
  // Render LaTeX to self-contained SVG at build time (zero client JS, SEO-friendly).
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathjax],
  },
  integrations: [
    starlight({
      title: 'triton-kernel-fused',
      description:
        'Drop-in fused Triton kernels (forward + backward) for transformer training on NVIDIA GPUs.',
      favicon: '/favicon.svg',
      tagline: 'Fused Triton kernels that beat torch.compile where it structurally can.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/IsNoobgrammer/triton-kernel-fused',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/IsNoobgrammer/triton-kernel-fused/edit/master/docs/',
      },
      sidebar: [
        {
          label: 'Get started',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'Quickstart', link: '/quickstart/' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'The structural edge', link: '/concepts/the-structural-edge/' },
            { label: 'Benchmarking', link: '/concepts/benchmarking/' },
          ],
        },
        {
          label: 'Kernels',
          items: [
            { label: 'Fused-linear cross-entropy', link: '/kernels/cross-entropy/' },
            { label: 'XSA correction', link: '/kernels/xsa/' },
            { label: 'Fused Muon', link: '/kernels/muon/' },
            { label: 'Conv MoE router', link: '/kernels/router/' },
            { label: 'PolyGLU MoE combine', link: '/kernels/moe/' },
          ],
        },
        {
          label: 'Contributing',
          items: [{ label: 'Contributing a kernel', link: '/contributing/' }],
        },
      ],
    }),
  ],
});
