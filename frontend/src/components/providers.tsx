'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { useEffect, useState } from 'react';
import { useThemeStore, useAuthStore } from '@/lib/store';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,
      refetchOnWindowFocus: false,
    },
  },
});

function ThemeProvider({ children }: { children: React.ReactNode }) {
  const { theme, setTheme } = useThemeStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    // Check system preference on mount
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('theme-storage');
    
    if (!savedTheme && systemPrefersDark) {
      setTheme('dark');
    } else {
      document.documentElement.classList.toggle('dark', theme === 'dark');
    }
  }, []);

  useEffect(() => {
    if (mounted) {
      document.documentElement.classList.toggle('dark', theme === 'dark');
    }
  }, [theme, mounted]);

  if (!mounted) return null;

  return <>{children}</>;
}

function AuthProvider({ children }: { children: React.ReactNode }) {
  const { checkAuth, isAuthenticated } = useAuthStore();

  useEffect(() => {
    // Initial auth check
    checkAuth();

    // Periodic auth check every 5 minutes to detect account deactivation
    const interval = setInterval(() => {
      if (isAuthenticated) {
        checkAuth();
      }
    }, 5 * 60 * 1000); // 5 minutes

    // Check auth when window regains focus
    const handleFocus = () => {
      if (isAuthenticated) {
        checkAuth();
      }
    };

    window.addEventListener('focus', handleFocus);

    return () => {
      clearInterval(interval);
      window.removeEventListener('focus', handleFocus);
    };
  }, [checkAuth, isAuthenticated]);

  return <>{children}</>;
}

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          {children}
          <Toaster
            position="top-right"
            containerStyle={{
              top: '80px', // Position below the navbar (adjust as needed)
            }}
            toastOptions={{
              duration: 4000,
              style: {
                background: 'var(--toast-bg)',
                color: 'var(--toast-color)',
                borderRadius: '12px',
                margin: '0 16px',
              },
              success: {
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#fff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
