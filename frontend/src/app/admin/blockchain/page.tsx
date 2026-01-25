'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  Shield, 
  ChevronDown, 
  ChevronRight,
  CheckCircle, 
  XCircle, 
  ExternalLink,
  RefreshCw,
  Search,
  User,
  Activity
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { blockchainApi } from '@/lib/api';
import { formatDateTime, truncateHash, getAnalysisTypeLabel, getRiskColor, getAnalysisTypeBgColor, getAnalysisTypeColor } from '@/lib/utils';
import Link from 'next/link';

interface BlockchainRecord {
  id: number;
  block_index: number;
  timestamp: string;
  data_hash: string;
  previous_hash: string;
  current_hash: string;
  analysis_id: string;
  analysis?: {
    id: string;
    analysis_type: string;
    risk_level: string;
    confidence: number;
    user?: {
      id: string;
      username: string;
      full_name: string;
    };
  };
}

export default function BlockchainPage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();
  const queryClient = useQueryClient();
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');

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

  // Fetch blockchain records
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['blockchain-records'],
    queryFn: () => blockchainApi.getRecords({ per_page: 100 }),
    enabled: isAuthenticated && user?.is_admin,
  });

  // Fetch chain validation
  const { data: validationData } = useQuery({
    queryKey: ['blockchain-validation'],
    queryFn: () => blockchainApi.validateChain(),
    enabled: isAuthenticated && user?.is_admin,
  });

  // Fetch stats
  const { data: statsData } = useQuery({
    queryKey: ['blockchain-stats'],
    queryFn: () => blockchainApi.getStats(),
    enabled: isAuthenticated && user?.is_admin,
  });

  const toggleRow = (index: number) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  const filteredRecords = data?.records?.filter((record: BlockchainRecord) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      record.current_hash.toLowerCase().includes(query) ||
      record.analysis?.user?.username?.toLowerCase().includes(query) ||
      record.analysis?.analysis_type?.toLowerCase().includes(query) ||
      (record.analysis?.analysis_type && getAnalysisTypeLabel(record.analysis.analysis_type).toLowerCase().includes(query))
    );
  });

  if (authLoading || !user?.is_admin) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-primary-100 dark:bg-primary-900/30">
              <Shield className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
              Blockchain Verification
            </h1>
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            Immutable audit trail of all medical analyses
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="card p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Total Blocks</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {statsData?.total_blocks || 0}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-blue-100 dark:bg-blue-900/30">
                <Activity className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
          </div>

          <div className="card p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Total Analyses</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {statsData?.total_analyses || 0}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-purple-100 dark:bg-purple-900/30">
                <Activity className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
          </div>

          <div className="card p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Chain Status</p>
                <div className="flex items-center gap-2 mt-1">
                  {validationData?.valid ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-500" />
                      <span className="text-lg font-semibold text-green-600">Valid</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="w-5 h-5 text-red-500" />
                      <span className="text-lg font-semibold text-red-600">Invalid</span>
                    </>
                  )}
                </div>
              </div>
              <div className={`p-3 rounded-xl ${validationData?.valid ? 'bg-green-100 dark:bg-green-900/30' : 'bg-red-100 dark:bg-red-900/30'}`}>
                <Shield className={`w-6 h-6 ${validationData?.valid ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`} />
              </div>
            </div>
          </div>

          <div className="card p-6">
            <button
              onClick={() => {
                queryClient.invalidateQueries({ queryKey: ['blockchain-records'] });
                queryClient.invalidateQueries({ queryKey: ['blockchain-validation'] });
                queryClient.invalidateQueries({ queryKey: ['blockchain-stats'] });
              }}
              className="w-full h-full flex flex-col items-center justify-center gap-2 text-slate-600 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
            >
              <RefreshCw className="w-8 h-8" />
              <span className="text-sm font-medium">Refresh Data</span>
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search by hash, username, or analysis type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input pl-12 w-full md:w-96"
            />
          </div>
        </div>

        {/* Blockchain Table */}
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 dark:bg-slate-800">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Block
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Hash
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Analysis Type
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    User
                  </th>
                  <th className="px-6 py-4 text-right text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Details
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                {isLoading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center">
                      <div className="flex items-center justify-center gap-3">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500" />
                        <span className="text-slate-500">Loading blockchain records...</span>
                      </div>
                    </td>
                  </tr>
                ) : filteredRecords?.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                      No blockchain records found
                    </td>
                  </tr>
                ) : (
                  filteredRecords?.map((record: BlockchainRecord) => (
                    <>
                      <tr
                        key={record.id}
                        className="hover:bg-slate-50 dark:hover:bg-slate-800/50 cursor-pointer transition-colors"
                        onClick={() => toggleRow(record.block_index)}
                      >
                        <td className="px-6 py-4">
                          <span className="inline-flex items-center px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-700 text-sm font-mono font-medium">
                            #{record.block_index}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-slate-600 dark:text-slate-400">
                          {formatDateTime(record.timestamp)}
                        </td>
                        <td className="px-6 py-4">
                          <code className="text-xs font-mono text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded">
                            {truncateHash(record.current_hash, 12)}
                          </code>
                        </td>
                        <td className="px-6 py-4">
                          {record.analysis ? (
                            <span 
                              className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${getAnalysisTypeBgColor(record.analysis.analysis_type)} ${getAnalysisTypeColor(record.analysis.analysis_type)}`}
                            >
                              {getAnalysisTypeLabel(record.analysis.analysis_type)}
                            </span>
                          ) : (
                            <span className="text-slate-400">-</span>
                          )}
                        </td>
                        <td className="px-6 py-4">
                          {record.analysis?.user ? (
                            <Link
                              href={`/admin/users/${record.analysis.user.id}`}
                              onClick={(e) => e.stopPropagation()}
                              className="text-primary-600 hover:text-primary-700 dark:text-primary-400 font-medium flex items-center gap-1"
                            >
                              <User className="w-4 h-4" />
                              {record.analysis.user.username}
                            </Link>
                          ) : (
                            <span className="text-slate-400">-</span>
                          )}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <button className="p-1 hover:bg-slate-100 dark:hover:bg-slate-700 rounded">
                            {expandedRows.has(record.block_index) ? (
                              <ChevronDown className="w-5 h-5 text-slate-500" />
                            ) : (
                              <ChevronRight className="w-5 h-5 text-slate-500" />
                            )}
                          </button>
                        </td>
                      </tr>
                      
                      {/* Expanded Details Row */}
                      <AnimatePresence>
                        {expandedRows.has(record.block_index) && (
                          <tr>
                            <td colSpan={6} className="px-0 py-0">
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden"
                              >
                                <div className="px-6 py-6 bg-slate-50 dark:bg-slate-800/50 border-t border-slate-100 dark:border-slate-700">
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Block Details */}
                                    <div>
                                      <h4 className="text-sm font-semibold text-slate-900 dark:text-white mb-4">
                                        Block Details
                                      </h4>
                                      <div className="space-y-3">
                                        <div>
                                          <span className="text-xs text-slate-500 dark:text-slate-400 block mb-1">
                                            Current Hash
                                          </span>
                                          <code className="text-xs font-mono text-slate-700 dark:text-slate-300 break-all">
                                            {record.current_hash}
                                          </code>
                                        </div>
                                        <div>
                                          <span className="text-xs text-slate-500 dark:text-slate-400 block mb-1">
                                            Previous Hash
                                          </span>
                                          <code className="text-xs font-mono text-slate-700 dark:text-slate-300 break-all">
                                            {record.previous_hash}
                                          </code>
                                        </div>
                                        <div>
                                          <span className="text-xs text-slate-500 dark:text-slate-400 block mb-1">
                                            Data Hash
                                          </span>
                                          <code className="text-xs font-mono text-slate-700 dark:text-slate-300 break-all">
                                            {record.data_hash}
                                          </code>
                                        </div>
                                      </div>
                                    </div>

                                    {/* Analysis Details */}
                                    {record.analysis && (
                                      <div>
                                        <h4 className="text-sm font-semibold text-slate-900 dark:text-white mb-4">
                                          Analysis Details
                                        </h4>
                                        <div className="space-y-3">
                                          <div className="flex items-center gap-2">
                                            <User className="w-4 h-4 text-slate-400" />
                                            <span className="text-sm text-slate-600 dark:text-slate-400">
                                              User:
                                            </span>
                                            {record.analysis.user ? (
                                              <Link
                                                href={`/admin/users/${record.analysis.user.id}`}
                                                className="text-sm text-primary-600 hover:underline font-medium"
                                              >
                                                {record.analysis.user.full_name} (@{record.analysis.user.username})
                                              </Link>
                                            ) : (
                                              <span className="text-sm text-slate-500">Unknown</span>
                                            )}
                                          </div>
                                          <div className="flex items-center gap-2">
                                            <Activity className="w-4 h-4 text-slate-400" />
                                            <span className="text-sm text-slate-600 dark:text-slate-400">
                                              Type:
                                            </span>
                                            <span className="text-sm font-medium text-slate-900 dark:text-white">
                                              {getAnalysisTypeLabel(record.analysis.analysis_type)}
                                            </span>
                                          </div>
                                          <div className="flex items-center gap-2">
                                            <Shield className="w-4 h-4 text-slate-400" />
                                            <span className="text-sm text-slate-600 dark:text-slate-400">
                                              Result:
                                            </span>
                                            <span className={`text-sm font-medium ${getRiskColor(record.analysis.risk_level || '')}`}>
                                              {record.analysis.risk_level || 'N/A'}
                                            </span>
                                          </div>
                                          <div className="flex items-center gap-2">
                                            <span className="text-sm text-slate-600 dark:text-slate-400">
                                              Confidence:
                                            </span>
                                            <span className="text-sm font-medium text-slate-900 dark:text-white">
                                              {record.analysis.confidence 
                                                ? `${(record.analysis.confidence * 100).toFixed(1)}%`
                                                : 'N/A'}
                                            </span>
                                          </div>
                                          <Link
                                            href={`/analysis/result/${record.analysis.id}`}
                                            className="inline-flex items-center gap-2 text-sm text-primary-600 hover:text-primary-700 font-medium mt-2"
                                          >
                                            View Full Analysis
                                            <ExternalLink className="w-4 h-4" />
                                          </Link>
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </motion.div>
                            </td>
                          </tr>
                        )}
                      </AnimatePresence>
                    </>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
