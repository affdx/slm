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

  // Webpack configuration
  webpack: (config, { isServer }) => {
    // Don't bundle onnxruntime-web on server side at all
    if (isServer) {
      config.externals = config.externals || [];
      config.externals.push('onnxruntime-web');
      config.externals.push('@mediapipe/tasks-vision');
    } else {
      // Client-side: mark these as external to use dynamic imports
      config.externals = {
        ...config.externals,
        'onnxruntime-node': 'commonjs onnxruntime-node',
      };
    }

    // Handle WASM files
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    // Don't process onnxruntime-web with Terser
    config.module.rules.push({
      test: /\.mjs$/,
      include: /node_modules\/onnxruntime-web/,
      type: 'javascript/auto',
    });

    return config;
  },

  // Transpile specific packages
  transpilePackages: [],

  // Headers for security and CORS - needed for SharedArrayBuffer
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'credentialless' },
        ],
      },
    ];
  },
};

export default nextConfig;
