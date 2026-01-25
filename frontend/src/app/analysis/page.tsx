'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Heart, Ribbon, Droplets, Scan, ArrowRight, Lock } from 'lucide-react';
import { useAuthStore } from '@/lib/store';

const analysisTypes = [
  {
    id: 'heart-prediction',
    title: 'Heart Risk Prediction',
    description: 'Heart risk prediction based on symptoms and risk factors',
    icon: Heart,
    color: 'red',
    href: '/analysis/heart-prediction',
    features: ['18 clinical parameters', 'ML-powered prediction', 'Risk score generation'],
  },
  {
    id: 'diabetes-prediction',
    title: 'Diabetes Risk Prediction',
    description: 'Diabetes risk prediction based on symptoms and lifestyle',
    icon: Droplets,
    color: 'orange',
    href: '/analysis/diabetes-prediction',
    features: ['Symptom analysis', 'Risk factor evaluation', 'Preventive recommendations'],
  },
  {
    id: 'skin-diagnosis',
    title: 'Skin Lesion Diagnosis',
    description: 'AI-powered skin lesion and condition analysis',
    icon: Scan,
    color: 'teal',
    href: '/analysis/skin-diagnosis',
    features: ['Image-based analysis', 'Multiple conditions', 'Confidence scoring'],
  },
  {
    id: 'breast-prediction',
    title: 'Breast Cancer Prediction',
    description: 'Clinical risk assessment using biomarkers and risk factors',
    icon: Ribbon,
    color: 'pink',
    href: '/analysis/breast-prediction',
    features: ['10 clinical parameters', 'Risk factor evaluation', 'Screening recommendations'],
  },
  {
    id: 'breast-diagnosis',
    title: 'Breast Tissue Diagnosis',
    description: 'Tissue-level tumor diagnosis using FNA measurements',
    icon: Ribbon,
    color: 'fuchsia',
    href: '/analysis/breast-diagnosis',
    features: ['30 cell nuclei features', 'Malignancy detection', '97.2% accuracy'],
  }
];

const colorClasses: Record<string, { bg: string; bgDark: string; iconBg: string; icon: string; border: string }> = {
  red: { 
    bg: 'bg-red-50', 
    bgDark: 'dark:bg-red-900/20', 
    iconBg: 'bg-red-100',
    icon: 'text-red-600 dark:text-red-400',
    border: 'border-red-200 dark:border-red-800'
  },
  pink: { 
    bg: 'bg-pink-50', 
    bgDark: 'dark:bg-pink-900/20', 
    iconBg: 'bg-pink-100',
    icon: 'text-pink-600 dark:text-pink-400',
    border: 'border-pink-200 dark:border-pink-800'
  },
  fuchsia: { 
    bg: 'bg-fuchsia-50', 
    bgDark: 'dark:bg-fuchsia-900/20', 
    iconBg: 'bg-fuchsia-100',
    icon: 'text-fuchsia-600 dark:text-fuchsia-400',
    border: 'border-fuchsia-200 dark:border-fuchsia-800'
  },
  orange: { 
    bg: 'bg-orange-50', 
    bgDark: 'dark:bg-orange-900/20', 
    iconBg: 'bg-orange-100',
    icon: 'text-orange-600 dark:text-orange-400',
    border: 'border-orange-200 dark:border-orange-800'
  },
  teal: { 
    bg: 'bg-teal-50', 
    bgDark: 'dark:bg-teal-900/20', 
    iconBg: 'bg-teal-100',
    icon: 'text-teal-600 dark:text-teal-400',
    border: 'border-teal-200 dark:border-teal-800'
  },
};

export default function AnalysisPage() {
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
            Medical Analysis Services
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Choose from our comprehensive suite of AI-powered diagnostic tools. 
            Each analysis uses advanced machine learning models for accurate predictions.
          </p>
          {!isAuthenticated && (
            <div className="mt-4 inline-flex items-center gap-2 text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 px-4 py-2 rounded-lg">
              <Lock className="w-5 h-5" />
              <span>Please <Link href="/login" className="underline font-medium">log in</Link> to use analysis features</span>
            </div>
          )}
        </motion.div>

        {/* Analysis Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {analysisTypes.map((type, index) => {
            const Icon = type.icon;
            const colors = colorClasses[type.color];
            
            return (
              <motion.div
                key={type.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link href={isAuthenticated ? type.href : '/login'}>
                  <div className={`card p-8 h-full hover:shadow-xl transition-all duration-300 group border-2 ${colors.border} ${colors.bg} ${colors.bgDark}`}>
                    {/* Icon */}
                    <div className={`inline-flex p-4 rounded-2xl ${colors.iconBg} ${colors.bgDark} shadow-md mb-6`}>
                      <Icon className={`w-10 h-10 ${colors.icon}`} />
                    </div>

                    {/* Title & Description */}
                    <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                      {type.title}
                    </h2>
                    <p className="text-slate-600 dark:text-slate-400 mb-6 leading-relaxed">
                      {type.description}
                    </p>

                    {/* Features */}
                    <div className="flex flex-wrap gap-2 mb-6">
                      {type.features.map((feature) => (
                        <span
                          key={feature}
                          className="px-3 py-1 text-sm bg-white dark:bg-slate-800 rounded-full text-slate-600 dark:text-slate-400 shadow-sm"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>

                    {/* CTA */}
                    <div className="flex items-center gap-2 text-primary-600 dark:text-primary-400 font-semibold group-hover:gap-3 transition-all">
                      <span>Start Analysis</span>
                      <ArrowRight className="w-5 h-5" />
                    </div>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>

        {/* Info Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <div className="card p-8 bg-gradient-to-r from-primary-50 to-secondary-50 dark:from-primary-900/20 dark:to-secondary-900/20 border-primary-200 dark:border-primary-800">
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-3">
              Secure & Confidential
            </h3>
            <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              All your medical data is encrypted and stored securely. Each analysis is recorded 
              on our blockchain for complete audit trail and data integrity verification.
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
