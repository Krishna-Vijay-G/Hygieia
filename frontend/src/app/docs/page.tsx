'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  ExternalLink,
  Heart,
  Droplets,
  Scan,
  Ribbon
} from 'lucide-react';

const modelDocs = [
  {
    id: 'heart-prediction',
    name: 'Heart Risk Prediction',
    accuracy: '99.36%',
    model: 'AdaBoost',
    description: 'Cardiovascular disease risk assessment',
    color: 'red',
    icon: Heart,
    features: ['18 clinical parameters', 'ML-powered prediction', 'Risk score generation'],
  },
  {
    id: 'diabetes-prediction',
    name: 'Diabetes Risk Prediction',
    accuracy: '98.1%',
    model: 'LightGBM',
    description: 'Early-stage diabetes risk screening',
    color: 'orange',
    icon: Droplets,
    features: ['Symptom analysis', 'Risk factor evaluation', 'Preventive recommendations'],
  },
  {
    id: 'skin-diagnosis',
    name: 'Skin Lesion Diagnosis',
    accuracy: '98.84%',
    model: 'CNN Ensemble',
    description: 'Multi-class skin condition classification',
    color: 'teal',
    icon: Scan,
    features: ['Image-based analysis', 'Multiple conditions', 'Confidence scoring'],
  },
  {
    id: 'breast-prediction',
    name: 'Breast Cancer Risk Prediction',
    accuracy: '81.3%',
    model: 'XGBoost Ensemble',
    description: 'Population-level breast cancer risk screening',
    color: 'pink',
    icon: Ribbon,
    features: ['10 clinical parameters', 'Risk factor evaluation', 'Screening recommendations'],
  },
  {
    id: 'breast-diagnosis',
    name: 'Breast Cancer Tissue Diagnosis',
    accuracy: '97.2%',
    model: 'Stacking Ensemble',
    description: 'FNA biopsy malignancy classification',
    color: 'fuchsia',
    icon: Ribbon,
    features: ['30 cell nuclei features', 'Malignancy detection', 'High accuracy diagnosis'],
  },
];

const colorClasses: Record<string, { bg: string; bgDark: string; iconBg: string; icon: string; border: string; text: string }> = {
  red: { 
    bg: 'bg-red-50', 
    bgDark: 'dark:bg-red-900/20', 
    iconBg: 'bg-red-100',
    icon: 'text-red-600 dark:text-red-400',
    border: 'border-red-200 dark:border-red-800',
    text: 'text-red-600 dark:text-red-400'
  },
  pink: { 
    bg: 'bg-pink-50', 
    bgDark: 'dark:bg-pink-900/20', 
    iconBg: 'bg-pink-100',
    icon: 'text-pink-600 dark:text-pink-400',
    border: 'border-pink-200 dark:border-pink-800',
    text: 'text-pink-600 dark:text-pink-400'
  },
  fuchsia: { 
    bg: 'bg-fuchsia-50', 
    bgDark: 'dark:bg-fuchsia-900/20', 
    iconBg: 'bg-fuchsia-100',
    icon: 'text-fuchsia-600 dark:text-fuchsia-400',
    border: 'border-fuchsia-200 dark:border-fuchsia-800',
    text: 'text-fuchsia-600 dark:text-fuchsia-400'
  },
  orange: { 
    bg: 'bg-orange-50', 
    bgDark: 'dark:bg-orange-900/20', 
    iconBg: 'bg-orange-100',
    icon: 'text-orange-600 dark:text-orange-400',
    border: 'border-orange-200 dark:border-orange-800',
    text: 'text-orange-600 dark:text-orange-400'
  },
  teal: { 
    bg: 'bg-teal-50', 
    bgDark: 'dark:bg-teal-900/20', 
    iconBg: 'bg-teal-100',
    icon: 'text-teal-600 dark:text-teal-400',
    border: 'border-teal-200 dark:border-teal-800',
    text: 'text-teal-600 dark:text-teal-400'
  },
};

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 py-12 px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-6xl mx-auto mb-16"
      >
        <h1 className="text-4xl sm:text-5xl font-bold text-slate-900 dark:text-white mb-4">
          Documentation
        </h1>
        <p className="text-xl text-slate-600 dark:text-slate-300 max-w-2xl">
          Complete guide to Hygieia AI Healthcare Platform. Learn about our advanced diagnostic models, API integration, and deployment best practices.
        </p>
      </motion.div>

      {/* AI Models Overview */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.5 }}
        className="max-w-6xl mx-auto mb-16"
      >
        <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-8">
          Diagnostic Models
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {modelDocs.map((model, index) => {
            const Icon = model.icon;
            const colors = colorClasses[model.color];
            
            return (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link href={`/docs/models/${model.id}`}>
                  <div className={`card p-8 h-full hover:shadow-xl transition-all duration-300 group border-2 ${colors.border} ${colors.bg} ${colors.bgDark}`}>
                    {/* Icon */}
                    <div className={`inline-flex p-4 rounded-2xl ${colors.iconBg} ${colors.bgDark} shadow-md mb-6`}>
                      <Icon className={`w-10 h-10 ${colors.icon}`} />
                    </div>

                    {/* Title & Description */}
                    <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                      {model.name}
                    </h3>
                    <p className="text-slate-600 dark:text-slate-400 mb-4 leading-relaxed">
                      {model.description}
                    </p>

                    {/* Model Type & Accuracy */}
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm font-medium text-slate-500 dark:text-slate-400">
                        {model.model}
                      </span>
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold text-white ${model.color === 'red' ? 'bg-red-500' : model.color === 'orange' ? 'bg-orange-500' : model.color === 'teal' ? 'bg-teal-500' : model.color === 'pink' ? 'bg-pink-500' : 'bg-fuchsia-500'}`}>
                        {model.accuracy}
                      </span>
                    </div>

                    {/* Features */}
                    <div className="flex flex-wrap gap-2 mb-6">
                      {model.features.map((feature) => (
                        <span
                          key={feature}
                          className="px-3 py-1 text-sm bg-white dark:bg-slate-800 rounded-full text-slate-600 dark:text-slate-400 shadow-sm"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>

                    {/* Link */}
                    <div className={`inline-flex items-center gap-2 text-sm font-semibold ${colors.text} group-hover:gap-3 transition-all`}>
                      View Documentation
                      <ExternalLink className="w-4 h-4" />
                    </div>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="max-w-4xl mx-auto mt-20 text-center"
      >
        <p className="text-slate-600 dark:text-slate-400 text-lg">
          For more information, visit our model documentation or contact support.
        </p>
      </motion.div>
    </div>
  );
}
