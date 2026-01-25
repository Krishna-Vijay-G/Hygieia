'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Brain,
  Heart,
  Scan,
  Ribbon,
  Droplet,
  Droplets,
  Activity,
  TrendingUp,
  Users,
  Award,
  Target,
  Zap,
  CheckCircle2,
  Box,
  Linkedin,
  Mail
} from 'lucide-react';
import Image from 'next/image';
import { useAuthStore } from '@/lib/store';

type TeamMember = {
  id: string;
  name: string;
  title: string;
  role: string;
  description: string;
  expertise: string[];
  image: string;
  linkedin: string;
  email: string;
};

type ModelInfo = {
  id: string;
  name: string;
  description?: string;
  icon?: string;
  colorClass?: string;
  darkColorClass?: string;
  bgClass?: string;
  borderClass?: string;
  activeBorder?: string;
  accuracy?: string;
  aucRoc?: string;
  dataset?: string;
  samples?: number;
  features?: number;
  architecture?: string;
  inferenceSpeed?: string;
  status?: string;
  href?: string;
  highlights?: string[];
};

type TeamData = {
  team: TeamMember[];
  stats: {
    modelsDeployed: number;
    totalAccuracy: string;
    patientsHelped: string;
    yearsExperience: string;
  };
  mission: string;
  vision: string;
  models?: ModelInfo[];
};

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      duration: 0.5
    }
  }
};

const iconMap: Record<string, any> = {
  Brain,
  Heart,
  Scan,
  Ribbon,
  Droplet,
  Droplets,
  Activity,
  TrendingUp,
  Users
};

// Models are loaded from /public/data/team.json. Removed hardcoded fallback to avoid duplication.

export default function AboutPage() {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [activeModel, setActiveModel] = useState<string>('');
  const { isAuthenticated } = useAuthStore();

  useEffect(() => {
    fetch('/data/team.json')
      .then(res => res.json())
      .then((data: TeamData) => {
        setTeamData(data);
        if (data.models && data.models.length) setActiveModel(data.models[0].id);
      })
      .catch(err => console.error('Failed to load team data:', err));
  }, []);

  // Helper component: tries to HEAD the image URL and only renders Next Image if it exists.
  function ValidatedImage({ src, alt, size }: { src?: string; alt: string; size?: number }) {
    const [exists, setExists] = useState<boolean>(false);

    useEffect(() => {
      let mounted = true;
      if (!src) {
        setExists(false);
        return;
      }
      // Try HEAD request to see if image is present in /public
      fetch(src, { method: 'HEAD' })
        .then(res => {
          if (!mounted) return;
          setExists(res.ok);
        })
        .catch(() => {
          if (!mounted) return;
          setExists(false);
        });
      return () => {
        mounted = false;
      };
    }, [src]);

    if (exists && src) {
      return (
        <Image src={src} alt={alt} width={size || 96} height={size || 96} className="object-cover w-24 h-24" />
      );
    }
    return <Users className="w-12 h-12 text-primary" />;
  }

  // Use JSON data from /public/data/team.json (no hardcoded fallback)
  const models = teamData?.models ?? [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Hero Section */}
      <section className="relative overflow-visible pt-20 pb-36">
        <div className="absolute inset-0 bg-grid-slate-100 dark:bg-grid-slate-800 [mask-image:linear-gradient(0deg,transparent,black)] dark:[mask-image:linear-gradient(0deg,transparent,white)]" />
        
        <div className="container mx-auto px-6 relative">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center max-w-4xl mx-auto"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 dark:bg-primary/20 rounded-full mb-6">
              <Brain className="w-5 h-5 text-primary" />
              <span className="text-sm font-medium text-primary">Hygieia AI Platform</span>
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-slate-900 via-blue-800 to-primary dark:from-white dark:via-blue-200 dark:to-primary bg-clip-text text-transparent">
              Advanced Medical AI Diagnostics
            </h1>
            
            <p className="text-xl text-slate-600 dark:text-slate-300 mb-8 leading-relaxed">
              Democratizing access to world-class medical diagnostics through cutting-edge machine learning,
              rigorous validation, and clinical expertise.
            </p>

            {/* Stats Grid */}
            {teamData && (
              <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="visible"
                className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-12"
              >
                {[
                  { label: 'Models Deployed', value: teamData.stats.modelsDeployed, icon: Box },
                  { label: 'Average Accuracy', value: teamData.stats.totalAccuracy, icon: Target },
                  { label: 'Patients Helped', value: teamData.stats.patientsHelped, icon: Users },
                  { label: 'Years Experience', value: teamData.stats.yearsExperience, icon: Award }
                ].map((stat, idx) => (
                  <motion.div
                    key={idx}
                    variants={itemVariants}
                    className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg border border-slate-200 dark:border-slate-700"
                  >
                    <stat.icon className="w-8 h-8 text-primary mb-3 mx-auto" />
                    <div className="text-3xl font-bold text-slate-900 dark:text-white mb-1">
                      {stat.value}
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      {stat.label}
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </motion.div>
        </div>
      </section>

      {/* Mission & Vision */}
      {teamData && (
        <section className="py-20 bg-white dark:bg-slate-900/50">
          <div className="container mx-auto px-6">
            <div className="grid md:grid-cols-2 gap-12 max-w-6xl mx-auto">
              <motion.div
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
                className="bg-gradient-to-br from-blue-50 to-primary/10 dark:from-blue-950/30 dark:to-primary/20 rounded-3xl p-8 border border-blue-200 dark:border-blue-800"
              >
                <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center mb-6">
                  <Target className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-4 text-slate-900 dark:text-white">Our Mission</h3>
                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                  {teamData.mission}
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
                className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/30 dark:to-pink-950/30 rounded-3xl p-8 border border-purple-200 dark:border-purple-800"
              >
                <div className="w-12 h-12 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl flex items-center justify-center mb-6">
                  <Zap className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-4 text-slate-900 dark:text-white">Our Vision</h3>
                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                  {teamData.vision}
                </p>
              </motion.div>
            </div>
          </div>
        </section>
      )}

      {/* Models Documentation */}
      <section className="py-20">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-4 text-slate-900 dark:text-white">
              Our AI Models
            </h2>
            <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              Five production-ready models trained on real medical data with rigorous validation
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
            {models.map((model, idx) => {
              const Icon = iconMap[model.icon || 'Users'] || Users;
              return (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                  onClick={() => setActiveModel(model.id)}
                  className={`bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-xl border-2 ${model.borderClass} transition-all cursor-pointer ${
                    activeModel === model.id ? `${model.activeBorder} scale-105 border-4` : ''
                  }`}
                >
                  <div className={`${model.bgClass} w-16 h-16 rounded-2xl flex items-center justify-center mb-4`}>
                    <Icon className={`w-8 h-8 ${model.colorClass} ${model.darkColorClass}`} />
                  </div>

                  <h3 className="text-xl font-bold mb-2 text-slate-900 dark:text-white">
                    {model.name}
                  </h3>

                  <div className="space-y-3 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Accuracy</span>
                      <span className="font-bold text-green-600 dark:text-green-400">{model.accuracy}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">AUC-ROC</span>
                      <span className="font-bold text-blue-600 dark:text-blue-400">{model.aucRoc}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Samples</span>
                      <span className="font-semibold text-slate-700 dark:text-slate-300">
                        {model.samples?.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Speed</span>
                      <span className="font-semibold text-slate-700 dark:text-slate-300">
                        {model.inferenceSpeed}
                      </span>
                    </div>
                  </div>

                  <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">Architecture</div>
                    <div className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      {model.architecture}
                    </div>
                  </div>

                  <div className="mt-4 inline-flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900/30 rounded-full">
                    <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400" />
                    <span className="text-xs font-medium text-green-700 dark:text-green-300">
                      {model.status}
                    </span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Team Section */}
      {teamData && (
        <section className="py-20 bg-slate-50 dark:bg-slate-900/50">
          <div className="container mx-auto px-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl font-bold mb-4 text-slate-900 dark:text-white">
                Meet Our Team
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
                Expert medical professionals, data scientists, and engineers dedicated to advancing healthcare AI
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
              {teamData.team.map((member, idx) => (
                <motion.div
                  key={member.id}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                  className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg border border-slate-200 dark:border-slate-700 hover:shadow-xl transition-shadow"
                >
                  <div className="w-24 h-24 rounded-full overflow-hidden mx-auto mb-4 flex items-center justify-center bg-gradient-to-br from-primary/20 to-blue-500/20">
                    {member.image ? (
                      <Image
                        src={member.image}
                        alt={member.name}
                        width={96}
                        height={96}
                        className="object-cover w-24 h-24"
                      />
                    ) : (
                      <Users className="w-12 h-12 text-primary" />
                    )}
                  </div>

                  <h3 className="text-xl font-bold text-center mb-1 text-slate-900 dark:text-white">
                    {member.name}
                  </h3>
                  <p className="text-sm text-primary font-medium text-center mb-2">
                    {member.title}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 text-center mb-4">
                    {member.role}
                  </p>

                  <p className="text-sm text-slate-600 dark:text-slate-300 text-center mb-4 leading-relaxed">
                    {member.description}
                  </p>

                  <div className="flex flex-wrap gap-2 justify-center mb-4">
                    {member.expertise.map((skill, skillIdx) => (
                      <span
                        key={skillIdx}
                        className="px-3 py-1 bg-primary/10 dark:bg-primary/20 text-primary text-xs rounded-full"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>

                  <div className="flex justify-center gap-3 pt-4 border-t border-slate-200 dark:border-slate-700">
                    <a
                      href={member.linkedin}
                      className="w-8 h-8 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center hover:bg-primary hover:text-white transition-colors"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Linkedin className="w-4 h-4" />
                    </a>
                    <a
                      href={`mailto:${member.email}`}
                      className="w-8 h-8 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center hover:bg-primary hover:text-white transition-colors"
                    >
                      <Mail className="w-4 h-4" />
                    </a>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Training Process Section */}
      <section className="py-20">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-4 text-slate-900 dark:text-white">
              Our Training Process
            </h2>
            <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              Rigorous methodology ensuring clinical-grade accuracy and reliability
            </p>
          </motion.div>

          <div className="max-w-5xl mx-auto">
            {[
              {
                step: 1,
                title: 'Data Collection & Curation',
                description: 'Large-scale medical datasets from trusted sources (UCI, BCSC, HAM10000) with comprehensive quality checks',
                icon: Box,
                color: 'blue'
              },
              {
                step: 2,
                title: 'Feature Engineering',
                description: 'Advanced feature extraction using deep learning embeddings, statistical analysis, and domain expertise',
                icon: Brain,
                color: 'purple'
              },
              {
                step: 3,
                title: 'Model Training & Optimization',
                description: 'Ensemble architectures with hyperparameter tuning, cross-validation, and class balancing techniques',
                icon: Activity,
                color: 'green'
              },
              {
                step: 4,
                title: 'Rigorous Validation',
                description: 'Multi-seed testing, confusion matrix analysis, and clinical performance metrics validation',
                icon: CheckCircle2,
                color: 'teal'
              },
              {
                step: 5,
                title: 'Production Deployment',
                description: 'Optimized inference, calibration tuning, and continuous monitoring for real-world performance',
                icon: Zap,
                color: 'amber'
              }
            ].map((process, idx) => (
              <motion.div
                key={process.step}
                initial={{ opacity: 0, x: idx % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: idx * 0.1 }}
                className="flex gap-6 mb-8 last:mb-0"
              >
                <div className="flex-shrink-0">
                  <div className={`w-16 h-16 bg-${process.color}-100 dark:bg-${process.color}-900/30 rounded-2xl flex items-center justify-center`}>
                    <process.icon className={`w-8 h-8 text-${process.color}-600 dark:text-${process.color}-400`} />
                  </div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`text-sm font-bold text-${process.color}-600 dark:text-${process.color}-400`}>
                      STEP {process.step}
                    </span>
                  </div>
                  <h3 className="text-xl font-bold mb-2 text-slate-900 dark:text-white">
                    {process.title}
                  </h3>
                  <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                    {process.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 bg-gradient-to-r from-primary to-blue-600">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center max-w-3xl mx-auto"
          >
            <h2 className="text-4xl font-bold mb-4 text-white">
              Ready to Experience AI-Powered Diagnostics?
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Join thousands of healthcare professionals using Hygieia for accurate, fast medical insights
            </p>
            <div className="flex gap-4 justify-center">
              {!isAuthenticated ? (
                <>
                  <a
                    href="/register"
                    className="px-8 py-4 bg-white text-primary rounded-xl font-semibold hover:bg-blue-50 transition-colors shadow-xl"
                  >
                    Get Started
                  </a>
                  <a
                    href="/docs"
                    className="px-8 py-4 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 transition-colors border-2 border-white"
                  >
                    API Documentation
                  </a>
                </>
              ) : (
                <>
                  <a
                    href="/dashboard"
                    className="px-8 py-4 bg-white text-primary rounded-xl font-semibold hover:bg-blue-50 transition-colors shadow-xl"
                  >
                    View Dashboard
                  </a>
                  <a
                    href="/docs"
                    className="px-8 py-4 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 transition-colors border-2 border-white"
                  >
                    API Documentation
                  </a>
                </>
              )}
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
