'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import Link from 'next/link';
import { 
  History, 
  Heart, 
  Ribbon, 
  Droplets, 
  Scan,
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  Activity,
  Shield
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { analysisApi } from '@/lib/api';
import { formatDateTime, getAnalysisTypeLabel, getRiskColor, getRiskIconBgColor, getRiskIconColor } from '@/lib/utils';

const analysisIcons: Record<string, any> = {
  'heart-prediction': Heart,
  'breast-prediction': Ribbon,
  'breast-diagnosis': Ribbon,
  'diabetes-prediction': Droplets,
  'skin-diagnosis': Scan,
};

const analysisColors: Record<string, { bg: string; icon: string }> = {
  'heart-prediction': { bg: 'bg-red-100 dark:bg-red-900/30', icon: 'text-red-600 dark:text-red-400' },
  'breast-prediction': { bg: 'bg-pink-100 dark:bg-pink-900/30', icon: 'text-pink-600 dark:text-pink-400' },
  'breast-diagnosis': { bg: 'bg-fuchsia-100 dark:bg-fuchsia-900/30', icon: 'text-fuchsia-600 dark:text-fuchsia-400' },
  'diabetes-prediction': { bg: 'bg-orange-100 dark:bg-orange-900/30', icon: 'text-orange-600 dark:text-orange-400' },
  'skin-diagnosis': { bg: 'bg-teal-100 dark:bg-teal-900/30', icon: 'text-teal-600 dark:text-teal-400' },
};

export default function HistoryPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [page, setPage] = useState(1);
  const perPage = 10;

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  const { data, isLoading } = useQuery({
    queryKey: ['analysis-history', page, filterType],
    queryFn: () => analysisApi.getHistory({ 
      page, 
      per_page: perPage,
      type: filterType === 'all' ? undefined : filterType 
    }),
    enabled: isAuthenticated,
  });

  const analyses = data?.analyses || [];
  const totalPages = Math.ceil((data?.total || 0) / perPage);

  const filteredAnalyses = analyses.filter((a: any) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      a.risk_level?.toLowerCase().includes(query) ||
      getAnalysisTypeLabel(a.analysis_type).toLowerCase().includes(query) ||
      a.model_name?.toLowerCase().includes(query) ||
      formatDateTime(a.created_at).toLowerCase().includes(query) ||
      (a.result && typeof a.result === 'object' && 
       JSON.stringify(a.result).toLowerCase().includes(query))
    );
  });

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-primary-100 dark:bg-primary-900/30">
              <History className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
              Analysis History
            </h1>
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            View all your past analyses and their results
          </p>
        </motion.div>

        {/* Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex flex-col sm:flex-row gap-4 mb-6"
        >
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search by result or type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input pl-12 w-full"
            />
          </div>
          <div className="relative">
            <Filter className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <select
              value={filterType}
              onChange={(e) => {
                setFilterType(e.target.value);
                setPage(1);
              }}
              className="input pl-12 pr-10 appearance-none"
            >
              <option value="all">All Types</option>
              <option value="heart-prediction">Heart Risk Prediction</option>
              <option value="diabetes-prediction">Diabetes Risk Prediction</option>
              <option value="skin-diagnosis">Skin Lesion Diagnosis</option>
              <option value="breast-prediction">Breast Risk Prediction</option>
              <option value="breast-diagnosis">Breast Tissue Diagnosis</option>
            </select>
          </div>
        </motion.div>

        {/* Analysis List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card overflow-hidden"
        >
          {isLoading ? (
            <div className="p-12 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto" />
            </div>
          ) : filteredAnalyses.length === 0 ? (
            <div className="p-12 text-center">
              <Activity className="w-16 h-16 text-slate-300 dark:text-slate-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
                No Analyses Found
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                {searchQuery || filterType !== 'all' 
                  ? 'Try adjusting your search or filters'
                  : "You haven't performed any analyses yet"}
              </p>
              <Link href="/analysis" className="btn btn-primary px-6">
                Start New Analysis
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-slate-100 dark:divide-slate-700">
              {filteredAnalyses.map((analysis: any, index: number) => {
                const Icon = analysisIcons[analysis.analysis_type] || Activity;
                const colors = analysisColors[analysis.analysis_type] || { 
                  bg: 'bg-slate-100 dark:bg-slate-800', 
                  icon: 'text-slate-600 dark:text-slate-400' 
                };

                return (
                  <Link
                    key={analysis.id}
                    href={`/analysis/result/${analysis.id}`}
                    className="block hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                  >
                    <div className="p-6 flex items-center gap-4">
                      <div className={`p-3 rounded-xl ${colors.bg}`}>
                        <Icon className={`w-6 h-6 ${colors.icon}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-slate-900 dark:text-white">
                          {getAnalysisTypeLabel(analysis.analysis_type)}
                          {analysis.blockchain_status === 'verified' && (
                            <Shield className="w-3.5 h-3.5 text-green-500 inline ml-2" />
                          )}
                          {analysis.blockchain_status === 'failed' && (
                            <Shield className="w-3.5 h-3.5 text-red-500 inline ml-2" />
                          )}
                        </h3>
                        <p className="text-sm text-slate-500 dark:text-slate-400">
                          {formatDateTime(analysis.created_at)}
                        </p>
                      </div>
                      <div className="text-right min-w-0">
                        <p className={`font-semibold ${analysis.risk_level ? getRiskColor(analysis.risk_level) : 'text-purple-600 dark:text-purple-400'}`}>
                          {analysis.risk_level || 'N/A'}
                        </p>
                        {analysis.confidence && (
                          <div className="mt-2 flex flex-col items-end">
                            <span className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                              {(analysis.confidence * 100).toFixed(0)}%
                            </span>
                            <div className="w-20 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-primary-500 rounded-full transition-all duration-300"
                                style={{ width: `${analysis.confidence * 100}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          )}
        </motion.div>

        {/* Pagination */}
        {totalPages > 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex items-center justify-between mt-6"
          >
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Showing {((page - 1) * perPage) + 1} to {Math.min(page * perPage, data?.total || 0)} of {data?.total || 0} results
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="btn btn-outline p-2 disabled:opacity-50"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <span className="px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="btn btn-outline p-2 disabled:opacity-50"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
