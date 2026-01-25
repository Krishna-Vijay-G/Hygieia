'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useMutation } from '@tanstack/react-query';
import { Scan, AlertCircle, Loader2, ArrowLeft, Upload, X, Image as ImageIcon } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { analysisApi } from '@/lib/api';
import Link from 'next/link';
import Image from 'next/image';

export default function SkinDiagnosisPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login?redirect=/analysis/skin-diagnosis');
    }
  }, [authLoading, isAuthenticated, router]);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }
    setError(null);
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const mutation = useMutation({
    mutationFn: async () => {
      if (!selectedFile) throw new Error('No file selected');
      return analysisApi.skinDiagnosis(selectedFile);
    },
    onSuccess: (data) => {
      router.push(`/analysis/result/${data.analysis_id}`);
    },
    onError: (error: any) => {
      setError(error.response?.data?.message || error.message || 'Analysis failed. Please try again.');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select an image to analyze');
      return;
    }
    setError(null);
    mutation.mutate();
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
          <div className="inline-flex p-4 rounded-2xl bg-teal-100 dark:bg-teal-900/30 mb-4">
            <Scan className="w-10 h-10 text-teal-600 dark:text-teal-400" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Skin Lesion Diagnosis
          </h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-xl mx-auto">
            Upload a clear image of the skin condition for AI-powered analysis. Our model can identify various skin conditions.
          </p>
        </motion.div>

        {/* Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-8"
        >
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Upload Area */}
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-4">
                Skin Image
              </label>
              
              {!preview ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-2xl p-12 text-center cursor-pointer hover:border-primary-400 dark:hover:border-primary-600 transition-colors"
                >
                  <div className="flex flex-col items-center gap-4">
                    <div className="p-4 rounded-full bg-slate-100 dark:bg-slate-800">
                      <Upload className="w-8 h-8 text-slate-400" />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-slate-700 dark:text-slate-300 mb-1">
                        Drop your image here or click to browse
                      </p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        Supports JPG, PNG, WebP • Max 10MB
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="relative">
                  <div className="relative w-full aspect-square max-w-md mx-auto rounded-2xl overflow-hidden bg-slate-100 dark:bg-slate-800">
                    <Image
                      src={preview}
                      alt="Preview"
                      fill
                      className="object-contain"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={clearFile}
                    className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                  <div className="mt-4 text-center">
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      <ImageIcon className="w-4 h-4 inline mr-1" />
                      {selectedFile?.name}
                    </p>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="text-sm text-primary-600 hover:text-primary-700 font-medium mt-2"
                    >
                      Change image
                    </button>
                  </div>
                </div>
              )}
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>

            {/* Guidelines */}
            <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-6">
              <h4 className="font-medium text-slate-900 dark:text-white mb-3">
                For Best Results:
              </h4>
              <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                <li className="flex items-start gap-2">
                  <span className="text-primary-500">•</span>
                  Use good lighting - natural light works best
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary-500">•</span>
                  Focus clearly on the affected area
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary-500">•</span>
                  Include some surrounding skin for context
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary-500">•</span>
                  Avoid blurry or overexposed images
                </li>
              </ul>
            </div>

            {/* Submit Button */}
            <div className="flex justify-end pt-4">
              <button
                type="submit"
                disabled={mutation.isPending || !selectedFile}
                className="btn btn-primary px-8 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {mutation.isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                    Analyzing Image...
                  </>
                ) : (
                  'Analyze Skin Condition'
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
            <strong>Note:</strong> This AI-powered analysis is for screening purposes only and cannot replace a professional dermatological examination. 
            Please consult a dermatologist for accurate diagnosis and treatment recommendations.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
