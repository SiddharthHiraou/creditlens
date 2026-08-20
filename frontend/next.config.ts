import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: the demo has to work as a plain link with no backend
  // running. Pages that want live scoring call the API from the browser and
  // fall back to the seeded snapshots when it is not reachable.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
