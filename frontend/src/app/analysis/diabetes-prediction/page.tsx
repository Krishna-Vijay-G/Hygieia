'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useMutation } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Droplets, AlertCircle, Loader2, ArrowLeft, Info } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { analysisApi } from '@/lib/api';
import Link from 'next/link';

const diabetesSchema = z.object({
  age: z.number().min(1).max(120),
  gender: z.string().min(1, "Please select a gender"),
  polyuria: z.enum(['Yes', 'No']),
  polydipsia: z.enum(['Yes', 'No']),
  sudden_weight_loss: z.enum(['Yes', 'No']),
  weakness: z.enum(['Yes', 'No']),
  polyphagia: z.enum(['Yes', 'No']),
  genital_thrush: z.enum(['Yes', 'No']),
  visual_blurring: z.enum(['Yes', 'No']),
  itching: z.enum(['Yes', 'No']),
  irritability: z.enum(['Yes', 'No']),
  delayed_healing: z.enum(['Yes', 'No']),
  partial_paresis: z.enum(['Yes', 'No']),
  muscle_stiffness: z.enum(['Yes', 'No']),
  alopecia: z.enum(['Yes', 'No']),
  obesity: z.enum(['Yes', 'No']),
});

type DiabetesFormData = z.infer<typeof diabetesSchema>;

const formFields = [
  { name: 'polyuria', label: 'Polyuria (Excessive Urination)', tooltip: 'Do you urinate more frequently than normal?' },
  { name: 'polydipsia', label: 'Polydipsia (Excessive Thirst)', tooltip: 'Do you experience excessive thirst?' },
  { name: 'sudden_weight_loss', label: 'Sudden Weight Loss', tooltip: 'Have you experienced unexplained weight loss?' },
  { name: 'weakness', label: 'Weakness', tooltip: 'Do you feel weak or fatigued?' },
  { name: 'polyphagia', label: 'Polyphagia (Excessive Hunger)', tooltip: 'Do you experience excessive hunger?' },
  { name: 'genital_thrush', label: 'Genital Thrush', tooltip: 'Have you experienced genital thrush infections?' },
  { name: 'visual_blurring', label: 'Visual Blurring', tooltip: 'Do you experience blurred vision?' },
  { name: 'itching', label: 'Itching', tooltip: 'Do you experience persistent itching?' },
  { name: 'irritability', label: 'Irritability', tooltip: 'Do you experience unusual irritability?' },
  { name: 'delayed_healing', label: 'Delayed Healing', tooltip: 'Do your wounds take longer to heal?' },
  { name: 'partial_paresis', label: 'Partial Paresis', tooltip: 'Do you experience partial loss of muscle movement?' },
  { name: 'muscle_stiffness', label: 'Muscle Stiffness', tooltip: 'Do you experience muscle stiffness?' },
  { name: 'alopecia', label: 'Alopecia (Hair Loss)', tooltip: 'Are you experiencing unusual hair loss?' },
  { name: 'obesity', label: 'Obesity', tooltip: 'Is your BMI above 30?' },
];

export default function DiabetesAnalysisPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login?redirect=/analysis/diabetes-prediction');
    }
  }, [authLoading, isAuthenticated, router]);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<DiabetesFormData>({
    resolver: zodResolver(diabetesSchema),
    defaultValues: {
      polyuria: 'No',
      polydipsia: 'No',
      sudden_weight_loss: 'No',
      weakness: 'No',
      polyphagia: 'No',
      genital_thrush: 'No',
      visual_blurring: 'No',
      itching: 'No',
      irritability: 'No',
      delayed_healing: 'No',
      partial_paresis: 'No',
      muscle_stiffness: 'No',
      alopecia: 'No',
      obesity: 'No',
    },
  });

  const mutation = useMutation({
    mutationFn: (data: DiabetesFormData) => {
      return analysisApi.diabetes(data);
    },
    onSuccess: (data) => {
      router.push(`/analysis/result/${data.analysis_id}`);
    },
    onError: (error: any) => {
      setError(error.response?.data?.message || 'Analysis failed. Please try again.');
    },
  });

  const onSubmit = (data: DiabetesFormData) => {
    setError(null);

    // Build a normalized payload with exact keys expected by the backend (snake_case)
    const payload = {
      age: Number(data.age),
      gender: data.gender,
      polyuria: data.polyuria,
      polydipsia: data.polydipsia,
      sudden_weight_loss: data.sudden_weight_loss,
      weakness: data.weakness,
      polyphagia: data.polyphagia,
      genital_thrush: data.genital_thrush,
      visual_blurring: data.visual_blurring,
      itching: data.itching,
      irritability: data.irritability,
      delayed_healing: data.delayed_healing,
      partial_paresis: data.partial_paresis,
      muscle_stiffness: data.muscle_stiffness,
      alopecia: data.alopecia,
      obesity: data.obesity,
    };

    mutation.mutate(payload);
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
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
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
          <div className="inline-flex p-4 rounded-2xl bg-orange-100 dark:bg-orange-900/30 mb-4">
            <Droplets className="w-10 h-10 text-orange-600 dark:text-orange-400" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Diabetes Risk Prediction
          </h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-xl mx-auto">
            Answer the following questions about your symptoms and health factors for early diabetes detection.
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

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-8 overflow-visible">
            {/* Basic Info */}
            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Basic Information
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Age
                  </label>
                  <input
                    type="number"
                    {...register('age', { valueAsNumber: true })}
                    className="input w-full"
                    placeholder="Enter your age"
                  />
                  {errors.age && (
                    <p className="mt-1 text-sm text-red-600">{errors.age.message}</p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Gender
                  </label>
                  <select {...register('gender')} className="input w-full">
                    <option value="">Select gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                  {errors.gender && (
                    <p className="mt-1 text-sm text-red-600">{errors.gender.message}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Symptoms */}
            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Symptoms & Health Factors
              </h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                Answer yes or no for each of the following symptoms:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 overflow-visible">
                {formFields.map((field) => (
                  <div
                    key={field.name}
                    className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl overflow-visible"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {field.label}
                      </span>
                      <div className="relative">
                        <div className="group">
                          <Info className="w-4 h-4 text-slate-400 cursor-help" />
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-slate-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none hidden group-hover:block">
                            {field.tooltip}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-3 items-center">
                      <button
                        type="button"
                        onClick={() => {
                          const newValue = watch(field.name as keyof DiabetesFormData) === 'Yes' ? 'No' : 'Yes';
                          setValue(field.name as keyof DiabetesFormData, newValue);
                        }}
                        className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                          watch(field.name as keyof DiabetesFormData) === 'Yes'
                            ? 'bg-primary-600'
                            : 'bg-slate-300 dark:bg-slate-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                            watch(field.name as keyof DiabetesFormData) === 'Yes'
                              ? 'translate-x-7'
                              : 'translate-x-1'
                          }`}
                        />
                      </button>
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {watch(field.name as keyof DiabetesFormData) === 'Yes' ? 'Yes' : 'No'}
                      </span>
                      <input
                        type="hidden"
                        value={watch(field.name as keyof DiabetesFormData)}
                        {...register(field.name as keyof DiabetesFormData)}
                      />
                    </div>
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
                  'Analyze Risk'
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
            <strong>Note:</strong> This analysis provides an early risk assessment based on the symptoms provided. 
            It is not a medical diagnosis. Please consult a healthcare professional for proper evaluation and diagnosis.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
