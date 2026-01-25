'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useMutation } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Ribbon, AlertCircle, Loader2, ArrowLeft, Info } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { analysisApi } from '@/lib/api';
import Link from 'next/link';

const breastCancerSchema = z.object({
  mean_radius: z.number().min(0).max(50),
  mean_texture: z.number().min(0).max(50),
  mean_perimeter: z.number().min(0).max(300),
  mean_area: z.number().min(0).max(3000),
  mean_smoothness: z.number().min(0).max(1),
});

type BreastCancerFormData = z.infer<typeof breastCancerSchema>;

const formFields = [
  { 
    name: 'mean_radius', 
    label: 'Mean Radius', 
    tooltip: 'Mean of distances from center to points on the perimeter (6-28 typical range)',
    placeholder: 'e.g., 14.5',
    min: 0,
    max: 50,
    step: 0.01,
  },
  { 
    name: 'mean_texture', 
    label: 'Mean Texture', 
    tooltip: 'Standard deviation of gray-scale values (9-40 typical range)',
    placeholder: 'e.g., 19.3',
    min: 0,
    max: 50,
    step: 0.01,
  },
  { 
    name: 'mean_perimeter', 
    label: 'Mean Perimeter', 
    tooltip: 'Mean size of the core tumor perimeter (40-190 typical range)',
    placeholder: 'e.g., 92.0',
    min: 0,
    max: 300,
    step: 0.001,
  },
  { 
    name: 'mean_area', 
    label: 'Mean Area', 
    tooltip: 'Mean area of the core tumor (140-2500 typical range)',
    placeholder: 'e.g., 655.0',
    min: 0,
    max: 3000,
    step: 0.001,
  },
  { 
    name: 'mean_smoothness', 
    label: 'Mean Smoothness', 
    tooltip: 'Local variation in radius lengths (0.05-0.16 typical range)',
    placeholder: 'e.g., 0.096',
    min: 0,
    max: 1,
    step: 0.00001,
  },
];

export default function BreastDiagnosisAnalysisPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login?redirect=/analysis/breast-diagnosis');
    }
  }, [authLoading, isAuthenticated, router]);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<BreastCancerFormData>({
    resolver: zodResolver(breastCancerSchema),
  });

  const mutation = useMutation({
    mutationFn: (data: BreastCancerFormData) => {
      return analysisApi.breastDiagnosis(data);
    },
    onSuccess: (data) => {
      router.push(`/analysis/result/${data.analysis_id}`);
    },
    onError: (error: any) => {
      setError(error.response?.data?.message || 'Analysis failed. Please try again.');
    },
  });

  const onSubmit = (data: BreastCancerFormData) => {
    setError(null);
    mutation.mutate(data);
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Back Link */}
        <Link
          href="/analysis"
          className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Analysis Types
        </Link>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="inline-flex p-4 rounded-2xl bg-fuchsia-100 dark:bg-fuchsia-900/30 mb-4">
            <Ribbon className="w-10 h-10 text-fuchsia-600 dark:text-fuchsia-400" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Breast Cancer Tissue Diagnosis
          </h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-xl mx-auto">
            Enter the fine needle aspirate (FNA) measurements from your clinical test results for tumor analysis.
          </p>
        </motion.div>

        {/* Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-8 overflow-visible"
        >
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6 overflow-visible">
            {/* Info Box */}
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-blue-700 dark:text-blue-300">
                  <p className="font-medium mb-1">About FNA Measurements</p>
                  <p>These values are typically obtained from a Fine Needle Aspirate (FNA) of a breast mass. 
                  The measurements describe characteristics of the cell nuclei present in the image.</p>
                </div>
              </div>
            </div>

            {/* Clinical Measurements */}
            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Cell Nuclei Measurements
              </h3>
              <div className="space-y-4">
                {formFields.map((field) => (
                  <div key={field.name}>
                    <div className="flex items-center gap-2 mb-2">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                        {field.label}
                      </label>
                      <div className="relative">
                        <div className="group">
                          <Info className="w-4 h-4 text-slate-400 cursor-help" />
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-slate-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none hidden group-hover:block">
                            {field.tooltip}
                          </div>
                        </div>
                      </div>
                    </div>
                    <input
                      type="number"
                      {...register(field.name as keyof BreastCancerFormData, { valueAsNumber: true })}
                      className="input w-full"
                      placeholder={field.placeholder}
                      min={field.min}
                      max={field.max}
                      step={field.step}
                    />
                    {errors[field.name as keyof BreastCancerFormData] && (
                      <p className="mt-1 text-sm text-red-600">
                        {errors[field.name as keyof BreastCancerFormData]?.message}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Submit Button */}
            <div className="flex justify-end pt-4">
              <button
                type="submit"
                disabled={mutation.isPending}
                className="btn btn-primary px-8 py-3"
              >
                {mutation.isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Tumor'
                )}
              </button>
            </div>
          </form>
        </motion.div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl"
        >
          <p className="text-sm text-amber-700 dark:text-amber-300">
            <strong>Note:</strong> This analysis uses machine learning to assess tumor characteristics. 
            Results are for screening purposes only and should be confirmed by a qualified healthcare professional through proper clinical diagnosis.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
