'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  Heart, 
  Ribbon,  
  Scan, 
  Droplets,
  ArrowRight, 
  Shield, 
  Zap, 
  Users,
  CheckCircle2,
  Sparkles
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';

const analysisTypes = [
  {
    id: 'heart-prediction',
    name: 'Heart Risk Prediction',
    description: 'Heart risk prediction based on symptoms and risk factors',
    icon: Heart,
    color: 'text-red-600',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
    iconBgColor: 'bg-red-100 dark:bg-red-900/20',
    borderColor: 'border-red-200 dark:border-red-800',
    accuracy: '99.4%',
    href: '/analysis/heart-prediction',
  },
  {
    id: 'diabetes-prediction',
    name: 'Diabetes Risk Prediction',
    description: 'Diabetes risk prediction based on symptoms and lifestyle',
    icon: Droplets,
    color: 'text-orange-600',
    bgColor: 'bg-orange-50 dark:bg-orange-900/20',
    iconBgColor: 'bg-orange-100 dark:bg-orange-900/20',
    borderColor: 'border-orange-200 dark:border-orange-800',
    accuracy: '98.1%',
    href: '/analysis/diabetes-prediction',
  },
  {
    id: 'skin-diagnosis',
    name: 'Skin Lesion Diagnosis',
    description: 'AI-powered skin lesion and condition analysis',
    icon: Scan,
    color: 'text-cyan-600',
    bgColor: 'bg-cyan-50 dark:bg-cyan-900/20',
    iconBgColor: 'bg-teal-100 dark:bg-cyan-900/20',
    borderColor: 'border-teal-200 dark:border-teal-800',
    accuracy: '96.8%',
    href: '/analysis/skin-diagnosis',
  },
  {
    id: 'breast-prediction',
    name: 'Breast Cancer Prediction',
    description: 'Clinical risk assessment using biomarkers and risk factors',
    icon: Ribbon,
    color: 'text-pink-600',
    bgColor: 'bg-pink-50 dark:bg-pink-900/20',
    iconBgColor: 'bg-pink-100 dark:bg-pink-900/20',
    borderColor: 'border-pink-200 dark:border-pink-800',
    accuracy: '81.3%',
    href: '/analysis/breast-prediction',
  },
  {
    id: 'breast-diagnosis',
    name: 'Breast Tissue Diagnosis',
    description: 'Tissue-level tumor diagnosis using FNA measurements',
    icon: Ribbon,
    color: 'text-fuchsia-600',
    bgColor: 'bg-fuchsia-50 dark:bg-fuchsia-900/20',
    iconBgColor: 'bg-fuchsia-100 dark:bg-fuchsia-900/20',
    borderColor: 'border-fuchsia-200 dark:border-fuchsia-800',
    accuracy: '97.2%',
    href: '/analysis/breast-diagnosis',
  },
];

const features = [
  {
    icon: Zap,
    title: 'Instant Results',
    description: 'Get comprehensive health assessments in seconds, powered by advanced ML models.',
  },
  {
    icon: Shield,
    title: 'Blockchain Verified',
    description: 'Every analysis is cryptographically secured and verifiable on our blockchain.',
  },
  {
    icon: Users,
    title: 'Expert Models',
    description: 'Trained on extensive medical datasets with accuracy rates exceeding 96%.',
  },
];

export default function HomePage() {
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="relative overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center">
        {/* Background effects */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-emerald-50 dark:from-slate-900 dark:via-slate-900 dark:to-emerald-950" />
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-30 dark:opacity-10" />
        
        {/* Floating shapes */}
        <motion.div
          className="absolute top-20 left-10 w-72 h-72 bg-primary-400/20 rounded-full blur-3xl"
          animate={{ y: [0, -20, 0], scale: [1, 1.1, 1] }}
          transition={{ duration: 8, repeat: Infinity }}
        />
        <motion.div
          className="absolute bottom-20 right-10 w-96 h-96 bg-emerald-400/20 rounded-full blur-3xl"
          animate={{ y: [0, 20, 0], scale: [1, 1.05, 1] }}
          transition={{ duration: 10, repeat: Infinity }}
        />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          {/* Mobile: Logo at top */}
          <div className="md:hidden flex justify-center mb-10">
            <motion.img
              src="/Hero.svg"
              alt="Hero image"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8 }}
              className="max-w-full w-[300px] sm:w-[380px]"
            />
          </div>

          <div className="grid md:grid-cols-2 items-center gap-10">
            {/* Left: content (existing) */}
            <div className="text-center md:text-left">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 text-sm font-medium mb-8">
                  <Sparkles className="w-4 h-4" />
                  AI-Powered Medical Diagnostics
                </span>
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="text-5xl md:text-6xl font-bold tracking-tight"
              >
                <span className="text-slate-900 dark:text-white">Your Health,</span>
                <br />
                <span className="gradient-text">Analyzed Intelligently</span>
              </motion.h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="mt-6 text-lg md:text-xl text-slate-600 dark:text-slate-400 max-w-2xl"
              >
                Advanced machine learning models delivering accurate health assessments 
                for cardiovascular, oncological, and dermatological conditions.
              </motion.p>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="mt-10 flex flex-col sm:flex-row items-center md:items-start justify-center md:justify-start gap-4"
              >
                {isAuthenticated ? (
                  <Link href="/dashboard" className="btn-primary btn-lg group">
                    Go to Dashboard
                    <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </Link>
                ) : (
                  <>
                    <Link href="/register" className="btn-primary btn-lg group">
                      Get Started Free
                      <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </Link>
                    <Link href="/login" className="btn-outline btn-lg">
                      Sign In
                    </Link>
                  </>
                )}
              </motion.div>

              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.6, delay: 0.5 }}
                className="mt-12 flex items-center gap-8 text-sm text-slate-500 dark:text-slate-400"
              >
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                  <span>99.4% Accuracy</span>
                </div>
                <div className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-blue-500" />
                  <span>HIPAA Compliant</span>
                </div>
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-yellow-500" />
                  <span>Startup Innovation</span>
                </div>
              </motion.div>
            </div>

            {/* Right: hero image (desktop only) */}
            <div className="hidden md:flex items-center justify-center md:justify-end">
              <motion.img
                src="/Hero.svg"
                alt="Hero image"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8 }}
                className="max-w-full w-[420px] md:w-[540px] lg:w-[640px]"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Analysis Types Section */}
      <section className="py-24 bg-white dark:bg-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="section-title">Comprehensive Health Analysis</h2>
            <p className="section-subtitle max-w-2xl mx-auto">
              Choose from our suite of AI-powered diagnostic tools, each trained on 
              extensive medical datasets for maximum accuracy.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6">
            {analysisTypes.map((type, index) => (
              <motion.div
                key={type.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Link href={type.href}>
                  <div className={`analysis-card ${type.bgColor} border ${type.borderColor} group`}>
                    <div className="flex items-start gap-4">
                      <div className={`p-3 rounded-xl ${type.iconBgColor}`}>
                        <type.icon className={`w-8 h-8 ${type.color}`} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                            {type.name}
                          </h3>
                          <span className="badge badge-success">{type.accuracy}</span>
                        </div>
                        <p className="text-slate-600 dark:text-slate-400">
                          {type.description}
                        </p>
                      </div>
                      <ArrowRight className="w-5 h-5 text-slate-400 group-hover:text-primary-500 group-hover:translate-x-1 transition-all" />
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-slate-50 dark:bg-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="text-center"
              >
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary-100 dark:bg-primary-900/30 mb-6">
                  <feature.icon className="w-8 h-8 text-primary-600 dark:text-primary-400" />
                </div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-br from-primary-600 to-emerald-600 relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
        
        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to Take Control of Your Health?
            </h2>
            <p className="text-xl text-primary-100 mb-10 max-w-2xl mx-auto">
              Join thousands of users who trust Hygieia for accurate, 
              AI-powered health assessments.
            </p>
            <Link 
              href={isAuthenticated ? '/dashboard' : '/register'} 
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-primary-600 font-semibold rounded-xl hover:bg-primary-50 transition-colors shadow-lg"
            >
              {isAuthenticated ? 'View Dashboard' : 'Create Free Account'}
              <ArrowRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
