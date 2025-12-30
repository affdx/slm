/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for containerized deployments (Sevalla, Docker)
  output: 'standalone',

  // Image optimization settings
  images: {
    // Allow images from these domains
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
    // Use default loader for standalone
    unoptimized: false,
  },

  // Environment variables available on client-side
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },

  // Headers for security and CORS
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,POST,OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
        ],
      },
    ];
  },

  // Rewrites to proxy API calls (optional alternative to API routes)
  // Uncomment if you prefer rewrites over API route proxying
  // async rewrites() {
  //   return [
  //     {
  //       source: '/backend/:path*',
  //       destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/:path*`,
  //     },
  //   ];
  // },
};

export default nextConfig;
