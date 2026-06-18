/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Emit a self-contained server bundle so the production Docker image
  // can run with just `node server.js` and no node_modules copy.
  output: "standalone",
};

export default nextConfig;
