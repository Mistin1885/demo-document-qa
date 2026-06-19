/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  // Let the API proxy preserve the exact path it receives. A framework-level
  // trailing-slash redirect can break streamed POST bodies before they reach
  // the backend proxy handler.
  skipTrailingSlashRedirect: true,
};

export default nextConfig;
