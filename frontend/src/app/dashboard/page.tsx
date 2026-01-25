'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import Link from 'next/link';
import { 
  Heart, 
  Ribbon, 
  Droplets, 
  Scan, 
  Activity, 
  TrendingUp,
  Clock,
  ArrowRight,
  Plus
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { analysisApi } from '@/lib/api';
import { formatDateTime, getAnalysisTypeLabel, getRiskColor } from '@/lib/utils';

const analysisTypes = [
  {
    id: 'heart-prediction',
    title: 'Heart Risk Prediction',
    description: 'Analyze cardiovascular health indicators',
    icon: Heart,
    color: 'red',
    href: '/analysis/heart-prediction',
  },
  {
    id: 'diabetes-prediction',
    title: 'Diabetes Risk Prediction',
    description: 'Assess diabetes risk factors',
    icon: Droplets,
    color: 'orange',
    href: '/analysis/diabetes-prediction',
  },
  {
    id: 'skin-diagnosis',
    title: 'Skin Lesion Diagnosis',
    description: 'Skin tissue and lesion analysis',
    icon: Scan,
    color: 'teal',
    href: '/analysis/skin-diagnosis',
  },
  {
    id: 'breast-prediction',
    title: 'Breast Cancer Prediction',
    description: 'Clinical risk assessment using biomarkers',
    icon: Ribbon,
    color: 'pink',
    href: '/analysis/breast-prediction',
  },
  {
    id: 'breast-diagnosis',
    title: 'Breast Tissue Diagnosis',
    description: 'Tissue-level tumor diagnosis using FNA measurements',
    icon: Ribbon,
    color: 'fuchsia',
    href: '/analysis/breast-diagnosis',
  },
];

const colorClasses: Record<string, { bg: string; icon: string }> = {
  red: { bg: 'bg-red-100 dark:bg-red-900/30', icon: 'text-red-600 dark:text-red-400' },
  pink: { bg: 'bg-pink-100 dark:bg-pink-900/30', icon: 'text-pink-600 dark:text-pink-400' },
  fuchsia: { bg: 'bg-fuchsia-100 dark:bg-fuchsia-900/30', icon: 'text-fuchsia-600 dark:text-fuchsia-400' },
  orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', icon: 'text-orange-600 dark:text-orange-400' },
  teal: { bg: 'bg-teal-100 dark:bg-teal-900/30', icon: 'text-teal-600 dark:text-teal-400' },
};

export default function DashboardPage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  // Fetch analysis history
  const { data: historyData, isLoading: historyLoading } = useQuery({
    queryKey: ['analysis-history'],
    queryFn: () => analysisApi.getHistory({ per_page: 5 }),
    enabled: isAuthenticated,
  });

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  const analysisHistory = historyData?.analyses || [];
  const analysisCounts = historyData?.counts || {};

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Welcome back, {user?.first_name}!
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Start a new analysis or review your previous results.
          </p>
        </motion.div>

        {/* Quick Stats: single Total card then five analysis-type boxes */}
        {/* Compute counts per analysis type */}
        {/* eslint-disable-next-line react-hooks/rules-of-hooks */}
        {(() => {
          const counts: Record<string, number> = {
            'heart-prediction': analysisCounts['heart-prediction'] || 0,
            'diabetes-prediction': analysisCounts['diabetes-prediction'] || 0,
            'skin-diagnosis': analysisCounts['skin-diagnosis'] || 0,
            'breast-prediction': analysisCounts['breast-prediction'] || 0,
            'breast-diagnosis': analysisCounts['breast-diagnosis'] || 0,
          };

          return (
            <div className="mb-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="card p-6 mb-4"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Total Analyses</p>
                    <p className="text-2xl font-bold text-slate-900 dark:text-white">
                      {historyData?.total || 0}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-primary-100 dark:bg-primary-900/30">
                    <Activity className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                  </div>
                </div>
              </motion.div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
                {/* Heart */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Heart</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['heart-prediction']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-red-100 dark:bg-red-900/30">
                      <Heart className="w-6 h-6 text-red-600 dark:text-red-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Diabetes */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Diabetes</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['diabetes-prediction']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-orange-100 dark:bg-orange-900/30">
                      <Droplets className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Skin */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.16 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Skin</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['skin-diagnosis']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-teal-100 dark:bg-teal-900/30">
                      <Scan className="w-6 h-6 text-teal-600 dark:text-teal-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Breast Risk (prediction) */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Breast Risk</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['breast-prediction']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-pink-100 dark:bg-pink-900/30">
                      <Ribbon className="w-6 h-6 text-pink-600 dark:text-pink-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Breast Tissue Diagnosis */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Breast Tissue</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['breast-diagnosis']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-fuchsia-100 dark:bg-fuchsia-900/30">
                      <Ribbon className="w-6 h-6 text-fuchsia-600 dark:text-fuchsia-400" />
                    </div>
                  </div>
                </motion.div>
              </div>
            </div>
          );
        })()}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Analysis Types */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-2"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                Start New Analysis
              </h2>
              <Link 
                href="/analysis" 
                className="text-primary-600 hover:text-primary-700 dark:text-primary-400 text-sm font-medium flex items-center gap-1"
              >
                View All
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {analysisTypes.map((type, index) => {
                const Icon = type.icon;
                const colors = colorClasses[type.color];
                return (
                  <motion.div
                    key={type.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + index * 0.05 }}
                  >
                    <Link href={type.href}>
                      <div className="card p-6 h-full hover:border-primary-300 dark:hover:border-primary-700 transition-all hover:shadow-lg group">
                        <div className={`inline-flex p-3 rounded-xl ${colors.bg} mb-4`}>
                          <Icon className={`w-6 h-6 ${colors.icon}`} />
                        </div>
                        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                          {type.title}
                        </h3>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          {type.description}
                        </p>
                      </div>
                    </Link>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>

          {/* Recent History */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                Recent Analyses
              </h2>
              <Link 
                href="/history" 
                className="text-primary-600 hover:text-primary-700 dark:text-primary-400 text-sm font-medium flex items-center gap-1"
              >
                View All
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
            <div className="card divide-y divide-slate-100 dark:divide-slate-700">
              {historyLoading ? (
                <div className="p-8 text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto" />
                </div>
              ) : analysisHistory.length === 0 ? (
                <div className="p-8 text-center text-slate-500 dark:text-slate-400">
                  <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No analyses yet</p>
                  <Link href="/analysis" className="text-primary-600 text-sm font-medium mt-2 inline-block">
                    Start your first analysis
                  </Link>
                </div>
              ) : (
                analysisHistory.map((analysis: any) => (
                  <Link
                    key={analysis.id}
                    href={`/analysis/result/${analysis.id}`}
                    className="block p-4 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-slate-900 dark:text-white">
                        {getAnalysisTypeLabel(analysis.analysis_type)}
                      </span>
                      <span className={`text-sm font-medium ${getRiskColor(analysis.risk_level || '')}`}>
                        {analysis.risk_level || 'N/A'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        {formatDateTime(analysis.created_at)}
                      </span>
                      {analysis.confidence && (
                        <span className="text-xs text-slate-500 dark:text-slate-400">
                          {(analysis.confidence * 100).toFixed(0)}% confidence
                        </span>
                      )}
                    </div>
                  </Link>
                ))
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
