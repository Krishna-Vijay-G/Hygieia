'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useMutation } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Heart, AlertCircle, Loader2, ArrowLeft, Info } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { analysisApi } from '@/lib/api';
import Link from 'next/link';

const heartDiseaseSchema = z.object({
  age: z.number().min(1).max(120),
  gender: z.string().min(1, "Please select a gender"),
  chest_pain: z.enum(['0', '1']),
  shortness_of_breath: z.enum(['0', '1']),
  fatigue: z.enum(['0', '1']),
  palpitations: z.enum(['0', '1']),
  dizziness: z.enum(['0', '1']),
  swelling: z.enum(['0', '1']),
  pain_arms_jaw_back: z.enum(['0', '1']),
  cold_sweats_nausea: z.enum(['0', '1']),
  high_bp: z.enum(['0', '1']),
  high_cholesterol: z.enum(['0', '1']),
  diabetes: z.enum(['0', '1']),
  smoking: z.enum(['0', '1']),
  obesity: z.enum(['0', '1']),
  sedentary_lifestyle: z.enum(['0', '1']),
  family_history: z.enum(['0', '1']),
  chronic_stress: z.enum(['0', '1']),
});

type HeartDiseaseFormData = z.infer<typeof heartDiseaseSchema>;

const formFields = [
  { name: 'chest_pain', label: 'Chest Pain', tooltip: 'Do you experience chest pain or discomfort?' },
  { name: 'shortness_of_breath', label: 'Shortness of Breath', tooltip: 'Do you experience difficulty breathing or shortness of breath?' },
  { name: 'fatigue', label: 'Fatigue', tooltip: 'Do you feel unusually tired or exhausted?' },
  { name: 'palpitations', label: 'Palpitations', tooltip: 'Do you notice irregular or rapid heartbeats?' },
  { name: 'dizziness', label: 'Dizziness', tooltip: 'Do you experience dizziness or lightheadedness?' },
  { name: 'swelling', label: 'Swelling', tooltip: 'Do you have swelling in legs, ankles, or feet?' },
  { name: 'pain_arms_jaw_back', label: 'Pain in Arms/Jaw/Back', tooltip: 'Do you experience pain radiating to arms, jaw, or back?' },
  { name: 'cold_sweats_nausea', label: 'Cold Sweats/Nausea', tooltip: 'Do you experience cold sweats or nausea?' },
  { name: 'high_bp', label: 'High Blood Pressure', tooltip: 'Have you been diagnosed with high blood pressure?' },
  { name: 'high_cholesterol', label: 'High Cholesterol', tooltip: 'Have you been diagnosed with high cholesterol?' },
  { name: 'diabetes', label: 'Diabetes', tooltip: 'Have you been diagnosed with diabetes?' },
  { name: 'smoking', label: 'Smoking', tooltip: 'Are you a current smoker?' },
  { name: 'obesity', label: 'Obesity', tooltip: 'Is your BMI above 30?' },
  { name: 'sedentary_lifestyle', label: 'Sedentary Lifestyle', tooltip: 'Do you have a sedentary lifestyle with little physical activity?' },
  { name: 'family_history', label: 'Family History', tooltip: 'Do you have a family history of heart disease?' },
  { name: 'chronic_stress', label: 'Chronic Stress', tooltip: 'Do you experience chronic stress?' },
];

export default function HeartDiseaseAnalysisPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login?redirect=/analysis/heart-prediction');
    }
  }, [authLoading, isAuthenticated, router]);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<HeartDiseaseFormData>({
    resolver: zodResolver(heartDiseaseSchema),
    defaultValues: {
      gender: '',
      chest_pain: '0',
      shortness_of_breath: '0',
      fatigue: '0',
      palpitations: '0',
      dizziness: '0',
      swelling: '0',
      pain_arms_jaw_back: '0',
      cold_sweats_nausea: '0',
      high_bp: '0',
      high_cholesterol: '0',
      diabetes: '0',
      smoking: '0',
      obesity: '0',
      sedentary_lifestyle: '0',
      family_history: '0',
      chronic_stress: '0',
    },
  });

  const mutation = useMutation({
    mutationFn: (data: HeartDiseaseFormData) => {
      const payload = {
        age: data.age,
        gender: parseInt(data.gender),
        chest_pain: parseInt(data.chest_pain),
        shortness_of_breath: parseInt(data.shortness_of_breath),
        fatigue: parseInt(data.fatigue),
        palpitations: parseInt(data.palpitations),
        dizziness: parseInt(data.dizziness),
        swelling: parseInt(data.swelling),
        pain_arms_jaw_back: parseInt(data.pain_arms_jaw_back),
        cold_sweats_nausea: parseInt(data.cold_sweats_nausea),
        high_bp: parseInt(data.high_bp),
        high_cholesterol: parseInt(data.high_cholesterol),
        diabetes: parseInt(data.diabetes),
        smoking: parseInt(data.smoking),
        obesity: parseInt(data.obesity),
        sedentary_lifestyle: parseInt(data.sedentary_lifestyle),
        family_history: parseInt(data.family_history),
        chronic_stress: parseInt(data.chronic_stress),
      };
      return analysisApi.heartDisease(payload);
    },
    onSuccess: (data) => {
      router.push(`/analysis/result/${data.analysis_id}`);
    },
    onError: (error: any) => {
      setError(error.response?.data?.message || 'Analysis failed. Please try again.');
    },
  });

  const onSubmit = (data: HeartDiseaseFormData) => {
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
          <div className="inline-flex p-4 rounded-2xl bg-red-100 dark:bg-red-900/30 mb-4">
            <Heart className="w-10 h-10 text-red-600 dark:text-red-400" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Heart Risk Prediction
          </h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-xl mx-auto">
            Complete the form below with accurate information for a comprehensive cardiovascular risk analysis.
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
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                  </select>
                  {errors.gender && (
                    <p className="mt-1 text-sm text-red-600">{errors.gender.message}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Symptoms & Risk Factors */}
            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Symptoms & Risk Factors
              </h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                Answer yes or no for each of the following:
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
                          const newValue = watch(field.name as keyof HeartDiseaseFormData) === '1' ? '0' : '1';
                          setValue(field.name as keyof HeartDiseaseFormData, newValue);
                        }}
                        className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                          watch(field.name as keyof HeartDiseaseFormData) === '1'
                            ? 'bg-primary-600'
                            : 'bg-slate-300 dark:bg-slate-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                            watch(field.name as keyof HeartDiseaseFormData) === '1'
                              ? 'translate-x-7'
                              : 'translate-x-1'
                          }`}
                        />
                      </button>
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {watch(field.name as keyof HeartDiseaseFormData) === '1' ? 'Yes' : 'No'}
                      </span>
                      <input
                        type="hidden"
                        value={watch(field.name as keyof HeartDiseaseFormData)}
                        {...register(field.name as keyof HeartDiseaseFormData)}
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
            <strong>Note:</strong> This analysis provides a risk assessment based on the information provided. 
            It is not a medical diagnosis. Please consult a healthcare professional for proper evaluation and diagnosis.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
