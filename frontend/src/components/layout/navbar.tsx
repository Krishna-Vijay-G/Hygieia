'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu, 
  X, 
  Sun, 
  Moon, 
  User, 
  LogOut, 
  Settings,
  LayoutDashboard,
  Shield,
  ChevronDown,
  MessageCircle
} from 'lucide-react';
import { useAuthStore, useThemeStore } from '@/lib/store';
import { cn, getImageURL } from '@/lib/utils';

const navigation = [
  { name: 'Home', href: '/' },
  { name: 'Analysis', href: '/analysis' },
  { name: 'Benchmarks', href: '/benchmarks' },
  { name: 'About', href: '/about' },
];

export function Navbar() {
  const pathname = usePathname();
  const { user, isAuthenticated, logout } = useAuthStore();
  const { theme, toggleTheme } = useThemeStore();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Close user menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setUserMenuOpen(false);
      }
    };

    if (userMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [userMenuOpen]);

  const handleLogout = async () => {
    await logout();
    setUserMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 glass border-b border-slate-200/50 dark:border-slate-700/50">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <img 
              src="/hygieia_color.png" 
              alt="Hygieia Logo" 
              className="h-10 w-auto"
            />
            <span className="text-xl font-bold text-slate-900 dark:text-white">
              Hygieia
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  pathname === item.href
                    ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                )}
              >
                {item.name}
              </Link>
            ))}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-2">
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5" />
              ) : (
                <Moon className="w-5 h-5" />
              )}
            </button>

            {/* Auth buttons / User menu */}
            {isAuthenticated ? (
              <div className="relative" ref={userMenuRef}>
                <button
                  onClick={() => setUserMenuOpen(!userMenuOpen)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                  {user?.avatar_url ? (
                    <img
                      src={getImageURL(user.avatar_url)}
                      alt={`${user?.first_name || 'User'} avatar`}
                      className="w-8 h-8 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-emerald-500 flex items-center justify-center">
                      <span className="text-white text-sm font-medium">
                        {user?.first_name?.[0]?.toUpperCase()}
                      </span>
                    </div>
                  )}
                  <span className="hidden sm:block text-sm font-medium text-slate-700 dark:text-slate-300">
                    {user?.first_name}
                  </span>
                  <ChevronDown className="w-4 h-4 text-slate-500" />
                </button>

                <AnimatePresence>
                  {userMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      className="absolute right-0 mt-2 w-56 rounded-xl bg-white dark:bg-slate-800 shadow-lg border border-slate-200 dark:border-slate-700 overflow-hidden"
                    >
                      <div className="p-3 border-b border-slate-100 dark:border-slate-700">
                        <p className="text-sm font-medium text-slate-900 dark:text-white">
                          {user?.full_name}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          {user?.email}
                        </p>
                      </div>
                      <div className="p-1">
                        <Link
                          href="/dashboard"
                          onClick={() => setUserMenuOpen(false)}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                        >
                          <LayoutDashboard className="w-4 h-4" />
                          Dashboard
                        </Link>
                        <Link
                          href="/profile"
                          onClick={() => setUserMenuOpen(false)}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                        >
                          <User className="w-4 h-4" />
                          Profile
                        </Link>
                        <Link
                          href="/settings"
                          onClick={() => setUserMenuOpen(false)}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                        >
                          <Settings className="w-4 h-4" />
                          Settings
                        </Link>
                        <div className="my-1 border-t border-slate-100 dark:border-slate-700" />
                        <Link
                          href="/chat"
                          onClick={() => setUserMenuOpen(false)}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                        >
                          <MessageCircle className="w-4 h-4" />
                          Dr. Hygieia
                          <span className="ml-auto text-xs bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 px-1.5 py-0.5 rounded">AI</span>
                        </Link>
                        {user?.is_admin && (
                          <>
                            <div className="my-1 border-t border-slate-100 dark:border-slate-700" />
                            <Link
                              href="/admin/blockchain"
                              onClick={() => setUserMenuOpen(false)}
                              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                            >
                              <Shield className="w-4 h-4" />
                              Blockchain
                            </Link>
                            <Link
                              href="/admin/users"
                              onClick={() => setUserMenuOpen(false)}
                              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                            >
                              <User className="w-4 h-4" />
                              Manage Users
                            </Link>
                          </>
                        )}
                      </div>
                      <div className="p-1 border-t border-slate-100 dark:border-slate-700">
                        <button
                          onClick={handleLogout}
                          className="flex w-full items-center gap-3 px-3 py-2 rounded-lg text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                        >
                          <LogOut className="w-4 h-4" />
                          Sign out
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <div className="hidden sm:flex items-center gap-2">
                <Link href="/login" className="btn-ghost btn-sm">
                  Sign in
                </Link>
                <Link href="/register" className="btn-primary btn-sm">
                  Get Started
                </Link>
              </div>
            )}

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-slate-200 dark:border-slate-700"
            >
              <div className="py-4 space-y-1">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    href={item.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className={cn(
                      'block px-4 py-2 rounded-lg text-base font-medium transition-colors',
                      pathname === item.href
                        ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400'
                        : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                    )}
                  >
                    {item.name}
                  </Link>
                ))}
                {!isAuthenticated && (
                  <div className="pt-4 border-t border-slate-200 dark:border-slate-700 space-y-2">
                    <Link
                      href="/login"
                      onClick={() => setMobileMenuOpen(false)}
                      className="block w-full btn-ghost btn-md text-center"
                    >
                      Sign in
                    </Link>
                    <Link
                      href="/register"
                      onClick={() => setMobileMenuOpen(false)}
                      className="block w-full btn-primary btn-md text-center"
                    >
                      Get Started
                    </Link>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* Click outside to close user menu */}
      {userMenuOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setUserMenuOpen(false)}
        />
      )}
    </header>
  );
}
