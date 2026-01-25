'use client';

import { } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { 
  BarChart3, 
  Heart, 
  Ribbon, 
  Droplets, 
  Scan,
  TrendingUp,
  Target,
  Zap,
  Clock
} from 'lucide-react';
import { benchmarkApi } from '@/lib/api';
import { formatDate, getAnalysisTypeLabel } from '@/lib/utils';

const analysisIcons: Record<string, any> = {
  'heart-prediction': Heart,
  'breast-prediction': Ribbon,
  'breast-diagnosis': Ribbon,
  'diabetes-prediction': Droplets,
  'skin-diagnosis': Scan,
};

const analysisColors: Record<string, { bg: string; icon: string; bar: string }> = {
  'heart-prediction': { 
    bg: 'bg-red-100 dark:bg-red-900/30', 
    icon: 'text-red-600 dark:text-red-400',
    bar: 'bg-red-500'
  },
  'diabetes-prediction': { 
    bg: 'bg-orange-100 dark:bg-orange-900/30', 
    icon: 'text-orange-600 dark:text-orange-400',
    bar: 'bg-orange-500'
  },
  'skin-diagnosis': { 
    bg: 'bg-teal-100 dark:bg-teal-900/30', 
    icon: 'text-teal-600 dark:text-teal-400',
    bar: 'bg-teal-500'
  },
  'breast-prediction': { 
    bg: 'bg-pink-100 dark:bg-pink-900/30', 
    icon: 'text-pink-600 dark:text-pink-400',
    bar: 'bg-pink-500'
  },
  'breast-diagnosis': { 
    bg: 'bg-fuchsia-100 dark:bg-fuchsia-900/30', 
    icon: 'text-fuchsia-600 dark:text-fuchsia-400',
    bar: 'bg-fuchsia-500'
  }
};

export default function BenchmarksPage() {
  

  const { data, isLoading } = useQuery({
    queryKey: ['benchmarks'],
    queryFn: () => benchmarkApi.getAll(),
    enabled: true,
  });

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  const benchmarks = data?.benchmarks || [];

  // Calculate averages
  const avgAccuracy = benchmarks.length > 0
    ? benchmarks.reduce((acc: number, b: any) => acc + (b.accuracy || 0), 0) / benchmarks.length
    : 0;
  const avgPrecision = benchmarks.length > 0
    ? benchmarks.reduce((acc: number, b: any) => acc + (b.precision_score || 0), 0) / benchmarks.length
    : 0;
  const avgRecall = benchmarks.length > 0
    ? benchmarks.reduce((acc: number, b: any) => acc + (b.recall || 0), 0) / benchmarks.length
    : 0;
  const avgF1 = benchmarks.length > 0
    ? benchmarks.reduce((acc: number, b: any) => acc + (b.f1_score || 0), 0) / benchmarks.length
    : 0;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-primary-100 dark:bg-primary-900/30">
              <BarChart3 className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
              Model Benchmarks
            </h1>
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            Performance metrics and accuracy statistics for all diagnostic models
          </p>
        </motion.div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Avg Accuracy</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {(avgAccuracy * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-xl bg-green-100 dark:bg-green-900/30">
                <Target className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Avg Precision</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {(avgPrecision * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-xl bg-blue-100 dark:bg-blue-900/30">
                <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Avg Recall</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {(avgRecall * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-xl bg-purple-100 dark:bg-purple-900/30">
                <TrendingUp className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="card p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">Avg F1 Score</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {(avgF1 * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-xl bg-amber-100 dark:bg-amber-900/30">
                <BarChart3 className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              </div>
            </div>
          </motion.div>
        </div>

        {/* Model Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {benchmarks.map((benchmark: any, index: number) => {
            const Icon = analysisIcons[benchmark.model_type] || BarChart3;
            const colors = analysisColors[benchmark.model_type] || { 
              bg: 'bg-slate-100 dark:bg-slate-800', 
              icon: 'text-slate-600 dark:text-slate-400',
              bar: 'bg-slate-500'
            };

            return (
              <motion.div
                key={benchmark.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className="card p-6"
              >
                {/* Header */}
                <div className="flex items-center gap-4 mb-6">
                  <div className={`p-3 rounded-xl ${colors.bg}`}>
                    <Icon className={`w-8 h-8 ${colors.icon}`} />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                      {benchmark.model_name || getAnalysisTypeLabel(benchmark.model_type)}
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      v{benchmark.model_version || '1.0.0'}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-slate-900 dark:text-white">
                      {(benchmark.accuracy * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">Accuracy</p>
                  </div>
                </div>

                {/* Metrics */}
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-600 dark:text-slate-400">Precision</span>
                      <span className="font-medium text-slate-900 dark:text-white">
                        {(benchmark.precision_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${colors.bar} rounded-full transition-all duration-500`}
                        style={{ width: `${benchmark.precision_score * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-600 dark:text-slate-400">Recall</span>
                      <span className="font-medium text-slate-900 dark:text-white">
                        {(benchmark.recall * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${colors.bar} rounded-full transition-all duration-500`}
                        style={{ width: `${benchmark.recall * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-600 dark:text-slate-400">F1 Score</span>
                      <span className="font-medium text-slate-900 dark:text-white">
                        {(benchmark.f1_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${colors.bar} rounded-full transition-all duration-500`}
                        style={{ width: `${benchmark.f1_score * 100}%` }}
                      />
                    </div>
                  </div>

                  {benchmark.auc_roc && (
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-600 dark:text-slate-400">AUC-ROC</span>
                        <span className="font-medium text-slate-900 dark:text-white">
                          {(benchmark.auc_roc * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${colors.bar} rounded-full transition-all duration-500`}
                          style={{ width: `${benchmark.auc_roc * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Footer */}
                <div className="mt-6 pt-4 border-t border-slate-100 dark:border-slate-700">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-500 dark:text-slate-400 flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      Test samples: {benchmark.test_samples?.toLocaleString() || 'N/A'}
                    </span>
                    {benchmark.created_at && (
                      <span className="text-slate-500 dark:text-slate-400">
                        {formatDate(benchmark.created_at)}
                      </span>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {benchmarks.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-12 text-center"
          >
            <BarChart3 className="w-16 h-16 text-slate-300 dark:text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              No Benchmarks Available
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              Model benchmarks will appear here once they are generated.
            </p>
          </motion.div>
        )}

        {/* Info Box */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 card p-6 bg-primary-50 dark:bg-primary-900/20 border-primary-200 dark:border-primary-800"
        >
          <h4 className="font-semibold text-slate-900 dark:text-white mb-2">
            Understanding the Metrics
          </h4>
          <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
            <li><strong>Accuracy:</strong> Overall correctness of the model predictions</li>
            <li><strong>Precision:</strong> Ratio of correct positive predictions to total positive predictions</li>
            <li><strong>Recall:</strong> Ratio of correct positive predictions to all actual positives</li>
            <li><strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
            <li><strong>AUC-ROC:</strong> Area under the Receiver Operating Characteristic curve</li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
}
