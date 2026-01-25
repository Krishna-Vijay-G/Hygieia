/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true,
  },
  // Allow cross-origin requests from devices on local network
  allowedDevOrigins: [
    'http://192.168.0.103:3000',
    'http://192.168.0.103:3001',
    'http://localhost:3000',
    'http://localhost:3001',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:3001',
  ],
}

module.exports = nextConfig
