'use client';

import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ShieldX, Home, AlertTriangle } from 'lucide-react';
import Link from 'next/link';
import { useAuthStore } from '@/lib/store';

export default function AccessDeniedPage() {
  const { isAuthenticated, isLoading } = useAuthStore();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      window.location.href = '/login';
    }
  }, [isAuthenticated, isLoading]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-md w-full"
      >
        <div className="card p-8 text-center">
          {/* Icon */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="mx-auto w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mb-6"
          >
            <ShieldX className="w-8 h-8 text-red-600 dark:text-red-400" />
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="text-2xl font-bold text-slate-900 dark:text-white mb-2"
          >
            Access Denied
          </motion.h1>

          {/* Description */}
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="text-slate-600 dark:text-slate-400 mb-8"
          >
            This page is only available to administrators. You don't have the required permissions to access this content.
          </motion.p>

          {/* Warning */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4 mb-8"
          >
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 mr-3 flex-shrink-0" />
              <p className="text-sm text-amber-800 dark:text-amber-200">
                If you believe this is an error, please contact your system administrator.
              </p>
            </div>
          </motion.div>

          {/* Actions */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="space-y-3"
          >
            <Link
              href="/dashboard"
              className="w-full btn-primary flex items-center justify-center"
            >
              <Home className="w-4 h-4 mr-2" />
              Go to Dashboard
            </Link>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}