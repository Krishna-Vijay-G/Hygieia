import Link from 'next/link';
import { Heart, Github, Linkedin, Globe } from 'lucide-react';

const footerLinks = {
  product: [
    { name: 'Analysis', href: '/analysis' },
    { name: 'Benchmarks', href: '/benchmarks' },
    { name: 'API Docs', href: '/docs' },
  ],
  company: [
    { name: 'About', href: '/about' },
    { name: 'Contact', href: '/contact' },
  ],
};

const socialLinks = [
  { name: 'Website', href: 'https://krishna-vijay-g.github.io', icon: Globe },
  { name: 'GitHub', href: 'https://github.com', icon: Github },
  { name: 'LinkedIn', href: 'https://linkedin.com/in/krishna-vijay', icon: Linkedin },
];

export function Footer() {
  return (
    <footer className="bg-slate-50 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-12 grid grid-cols-2 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-2">
            <Link href="/" className="flex items-center gap-2 mb-4">
              <img 
                src="/hygieia_color.png" 
                alt="Hygieia Logo" 
                className="h-10 w-auto"
              />
              <span className="text-xl font-bold text-slate-900 dark:text-white">
                Hygieia
              </span>
            </Link>
            <p className="text-slate-600 dark:text-slate-400 text-sm max-w-xs mb-4">
              AI-powered medical diagnostics platform delivering accurate health 
              assessments with blockchain verification.
            </p>
            <div className="flex items-center gap-4">
              {socialLinks.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                >
                  <item.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Links */}
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white mb-4">
              Product
            </h3>
            <ul className="space-y-3">
              {footerLinks.product.map((link) => (
                <li key={link.name}>
                  <Link
                    href={link.href}
                    className="text-sm text-slate-600 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white mb-4">
              Company
            </h3>
            <ul className="space-y-3">
              {footerLinks.company.map((link) => (
                <li key={link.name}>
                  <Link
                    href={link.href}
                    className="text-sm text-slate-600 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="py-6 border-t border-slate-200 dark:border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            © {new Date().getFullYear()} Hygieia. All rights reserved.
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 flex items-center gap-1">
            Made with <Heart className="w-4 h-4 text-red-500 fill-red-500" /> by Team Hygieia
          </p>
        </div>
      </div>
    </footer>
  );
}
