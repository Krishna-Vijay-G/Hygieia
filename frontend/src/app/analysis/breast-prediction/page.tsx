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

const breastPredictionSchema = z.object({
  actual_age: z.number().min(18).max(120),
  actual_bmi: z.number().min(10).max(60),
  density: z.number().min(1).max(4),
  race: z.number().min(1).max(6),
  age_menarche: z.number().min(0).max(2),
  age_first_birth: z.number().min(0).max(4),
  brstproc: z.number().min(0).max(1),
  hrt: z.number().min(0).max(1),
  family_hx: z.number().min(0).max(1),
  menopaus: z.number().min(1).max(3),
});

type BreastPredictionFormData = z.infer<typeof breastPredictionSchema>;

// Helper function to convert actual age to age group
const convertAgeToGroup = (age: number): number => {
  if (age >= 18 && age <= 29) return 1;
  if (age >= 30 && age <= 34) return 2;
  if (age >= 35 && age <= 39) return 3;
  if (age >= 40 && age <= 44) return 4;
  if (age >= 45 && age <= 49) return 5;
  if (age >= 50 && age <= 54) return 6;
  if (age >= 55 && age <= 59) return 7;
  if (age >= 60 && age <= 64) return 8;
  if (age >= 65 && age <= 69) return 9;
  if (age >= 70 && age <= 74) return 10;
  if (age >= 75 && age <= 79) return 11;
  if (age >= 80 && age <= 84) return 12;
  return 13; // 85+
};

// Helper function to convert actual BMI to BMI category
const convertBMIToCategory = (bmi: number): number => {
  if (bmi >= 10 && bmi < 25) return 1;
  if (bmi >= 25 && bmi < 30) return 2;
  if (bmi >= 30 && bmi < 35) return 3;
  return 4; // 35+
};

const formFields = [
  { 
    name: 'actual_age', 
    label: 'Age', 
    tooltip: 'Your current age in years',
    placeholder: 'e.g., 55',
    min: 18,
    max: 120,
    step: 1,
    unit: 'years',
  },
  { 
    name: 'actual_bmi', 
    label: 'Body Mass Index (BMI)', 
    tooltip: 'Your BMI = weight(kg) / height(m)². Online calculators available.',
    placeholder: 'e.g., 27.5',
    min: 10,
    max: 60,
    step: 0.1,
    unit: 'kg/m²',
  },
  { 
    name: 'density', 
    label: 'Breast Density (BI-RADS)', 
    tooltip: 'From your mammogram report',
    placeholder: 'Select density',
    min: 1,
    max: 4,
    step: 1,
    options: [
      { value: 1, label: 'Almost entirely fat' },
      { value: 2, label: 'Scattered fibroglandular densities' },
      { value: 3, label: 'Heterogeneously dense' },
      { value: 4, label: 'Extremely dense' },
    ],
  },
  { 
    name: 'race', 
    label: 'Race/Ethnicity', 
    tooltip: 'Select your race/ethnicity',
    placeholder: 'Select race/ethnicity',
    min: 1,
    max: 6,
    step: 1,
    options: [
      { value: 1, label: 'Non-Hispanic White' },
      { value: 2, label: 'Non-Hispanic Black' },
      { value: 3, label: 'Asian/Pacific Islander' },
      { value: 4, label: 'Native American' },
      { value: 5, label: 'Hispanic' },
      { value: 6, label: 'Other/Mixed' },
    ],
  },
  { 
    name: 'age_menarche', 
    label: 'Age at First Menstrual Period', 
    tooltip: 'How old were you when you had your first period?',
    placeholder: 'Select age range',
    min: 0,
    max: 2,
    step: 1,
    options: [
      { value: 0, label: 'Age 14 or older' },
      { value: 1, label: 'Age 12-13' },
      { value: 2, label: 'Age younger than 12' },
    ],
  },
  { 
    name: 'age_first_birth', 
    label: 'Age at First Live Birth', 
    tooltip: 'How old were you when you had your first child?',
    placeholder: 'Select age range',
    min: 0,
    max: 4,
    step: 1,
    options: [
      { value: 0, label: 'Younger than 20' },
      { value: 1, label: 'Age 20-24' },
      { value: 2, label: 'Age 25-29' },
      { value: 3, label: 'Age 30 or older' },
      { value: 4, label: 'No children (Nulliparous)' },
    ],
  },
  { 
    name: 'brstproc', 
    label: 'Previous Breast Biopsy', 
    tooltip: 'Have you had a breast biopsy or aspiration?',
    placeholder: 'Select Yes or No',
    min: 0,
    max: 1,
    step: 1,
    options: [
      { value: 0, label: 'No' },
      { value: 1, label: 'Yes' },
    ],
  },
  { 
    name: 'hrt', 
    label: 'Hormone Replacement Therapy (HRT)', 
    tooltip: 'Are you currently using hormone replacement therapy?',
    placeholder: 'Select Yes or No',
    min: 0,
    max: 1,
    step: 1,
    options: [
      { value: 0, label: 'No' },
      { value: 1, label: 'Yes' },
    ],
  },
  { 
    name: 'family_hx', 
    label: 'Family History of Breast Cancer', 
    tooltip: 'Has a first-degree relative (mother, sister, daughter) had breast cancer?',
    placeholder: 'Select Yes or No',
    min: 0,
    max: 1,
    step: 1,
    options: [
      { value: 0, label: 'No' },
      { value: 1, label: 'Yes' },
    ],
  },
  { 
    name: 'menopaus', 
    label: 'Menopausal Status', 
    tooltip: 'What is your current menopausal status?',
    placeholder: 'Select status',
    min: 1,
    max: 3,
    step: 1,
    options: [
      { value: 1, label: 'Premenopausal or Perimenopausal' },
      { value: 2, label: 'Postmenopausal' },
      { value: 3, label: 'Surgical menopause' },
    ],
  },
];

export default function BreastPredictionPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login?redirect=/analysis/breast-prediction');
    }
  }, [authLoading, isAuthenticated, router]);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<BreastPredictionFormData>({
    resolver: zodResolver(breastPredictionSchema),
  });

  const mutation = useMutation({
    mutationFn: (data: BreastPredictionFormData) => {
      // Helper function to get label from options
      const getLabel = (fieldName: string, value: number): string => {
        const field = formFields.find(f => f.name === fieldName);
        const option = field?.options?.find(opt => opt.value === value);
        return option?.label || String(value);
      };

      // Convert user-friendly values to model-coded values
      const convertedData = {
        age: convertAgeToGroup(data.actual_age),
        bmi: convertBMIToCategory(data.actual_bmi),
        density: data.density,
        race: data.race,
        agefirst: data.age_menarche,
        nrelbc: data.age_first_birth,
        brstproc: data.brstproc,
        hrt: data.hrt,
        family_hx: data.family_hx,
        menopaus: data.menopaus,
        // Also send original user-friendly values for display
        actual_age: data.actual_age,
        actual_bmi: data.actual_bmi,
        actual_density: getLabel('density', data.density),
        actual_race: getLabel('race', data.race),
        actual_age_menarche: getLabel('age_menarche', data.age_menarche),
        actual_age_first_birth: getLabel('age_first_birth', data.age_first_birth),
        actual_brstproc: getLabel('brstproc', data.brstproc),
        actual_hrt: getLabel('hrt', data.hrt),
        actual_family_hx: getLabel('family_hx', data.family_hx),
        actual_menopaus: getLabel('menopaus', data.menopaus),
      };
      return analysisApi.breastPrediction(convertedData);
    },
    onSuccess: (data) => {
      router.push(`/analysis/result/${data.analysis_id}`);
    },
    onError: (error: any) => {
      setError(error.response?.data?.message || 'Analysis failed. Please try again.');
    },
  });

  const onSubmit = (data: BreastPredictionFormData) => {
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
          <div className="inline-flex p-4 rounded-2xl bg-pink-100 dark:bg-pink-900/30 mb-4">
            <Ribbon className="w-10 h-10 text-pink-600 dark:text-pink-400" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Breast Cancer Risk Prediction
          </h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-xl mx-auto">
            Enter clinical indicators and biomarker levels to assess breast cancer risk based on metabolic and hormonal factors.
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
                  <p className="font-medium mb-1">About Breast Cancer Risk Prediction</p>
                  <p>This assessment uses the BCSC (Breast Cancer Surveillance Consortium) model to evaluate 
                  your breast cancer risk based on clinical risk factors. All information is confidential.</p>
                </div>
              </div>
            </div>

            {/* Clinical Parameters */}
            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Clinical Risk Factors
              </h3>
              <div className="space-y-4">
                {formFields.map((field) => (
                  <div key={field.name}>
                    <div className="flex items-center gap-2 mb-2">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                        {field.label}
                        {field.unit && <span className="text-slate-500 ml-1">({field.unit})</span>}
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
                    
                    {field.options ? (
                      <select
                        {...register(field.name as keyof BreastPredictionFormData, { valueAsNumber: true })}
                        className="input w-full"
                      >
                        <option value="">{field.placeholder}</option>
                        {field.options.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        {...register(field.name as keyof BreastPredictionFormData, { valueAsNumber: true })}
                        className="input w-full"
                        placeholder={field.placeholder}
                        min={field.min}
                        max={field.max}
                        step={field.step}
                      />
                    )}
                    
                    {errors[field.name as keyof BreastPredictionFormData] && (
                      <p className="mt-1 text-sm text-red-600">
                        {errors[field.name as keyof BreastPredictionFormData]?.message}
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
                  'Assess Risk'
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
            <strong>Note:</strong> This risk assessment uses machine learning to analyze clinical biomarkers. 
            Results are for screening purposes only and should not replace regular mammography or clinical breast examinations. 
            Consult with your healthcare provider for proper screening recommendations.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
