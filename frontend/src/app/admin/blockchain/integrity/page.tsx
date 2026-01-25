'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { Shield, CheckCircle, XCircle, AlertTriangle, ArrowLeft } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { blockchainApi } from '@/lib/api';
import Link from 'next/link';

export default function BlockchainIntegrityPage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();

  // Check auth and admin status
  useEffect(() => {
    if (!authLoading) {
      if (!isAuthenticated) {
        router.push('/login');
      } else if (!user?.is_admin) {
        router.push('/access-denied');
      }
    }
  }, [authLoading, isAuthenticated, user, router]);

  // Fetch chain validation
  const { data: validationData, isLoading, refetch } = useQuery({
    queryKey: ['blockchain-validation'],
    queryFn: () => blockchainApi.validateChain(),
    enabled: isAuthenticated && user?.is_admin,
  });

  if (authLoading || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  if (!isAuthenticated || !user?.is_admin) {
    return null;
  }

  const isValid = validationData?.is_valid;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Back Link */}
        <Link
          href="/admin/blockchain"
          className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Blockchain Records
        </Link>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-primary-100 dark:bg-primary-900/30">
              <Shield className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
              Blockchain Integrity Check
            </h1>
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            Validate the integrity and security of the blockchain
          </p>
        </motion.div>

        {/* Validation Result */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className={`card p-8 border-2 ${
            isValid
              ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20'
              : 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20'
          }`}
        >
          <div className="flex items-center gap-4">
            <div className={`p-4 rounded-full ${
              isValid
                ? 'bg-green-100 dark:bg-green-900/50'
                : 'bg-red-100 dark:bg-red-900/50'
            }`}>
              {isValid ? (
                <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
              ) : (
                <XCircle className="w-8 h-8 text-red-600 dark:text-red-400" />
              )}
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
                {isValid ? 'Blockchain is Valid' : 'Integrity Issues Detected'}
              </h2>
              <p className={`text-lg ${
                isValid
                  ? 'text-green-700 dark:text-green-300'
                  : 'text-red-700 dark:text-red-300'
              }`}>
                {isValid
                  ? 'All blocks are properly linked and verified'
                  : 'Some blocks have integrity issues that require attention'}
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400">Total Blocks</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {validationData?.total_blocks || 0}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400">Invalid Blocks</p>
                <p className={`text-2xl font-bold ${
                  validationData?.invalid_blocks?.length > 0
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-green-600 dark:text-green-400'
                }`}>
                  {validationData?.invalid_blocks?.length || 0}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400">Status</p>
                <p className={`text-2xl font-bold ${
                  isValid
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-red-600 dark:text-red-400'
                }`}>
                  {isValid ? 'SECURE' : 'AT RISK'}
                </p>
              </div>
            </div>
          </div>

          {/* Invalid Blocks List */}
          {validationData?.invalid_blocks?.length > 0 && (
            <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Invalid Blocks
              </h3>
              <div className="space-y-2">
                {validationData.invalid_blocks.map((blockIndex: number) => (
                  <div
                    key={blockIndex}
                    className="flex items-center gap-3 p-3 bg-white dark:bg-slate-800 rounded-lg"
                  >
                    <AlertTriangle className="w-5 h-5 text-red-500" />
                    <span className="text-slate-900 dark:text-white">
                      Block #{blockIndex}
                    </span>
                    <span className="text-sm text-slate-500 dark:text-slate-400">
                      Hash mismatch detected
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-6 flex gap-4"
        >
          <button
            onClick={() => refetch()}
            className="btn btn-primary"
          >
            Re-validate Chain
          </button>
          <Link href="/admin/blockchain" className="btn btn-outline">
            View All Records
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
