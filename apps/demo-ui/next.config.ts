import type { NextConfig } from "next";

// The FastAPI backend (`uv run pv-demo`). The browser only ever talks to this app on :3000.
const API_URL = process.env.PV_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Compression buffers the server-sent event stream, so the live feed would stall.
  compress: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
