'use client';

import { useRouter } from 'next/navigation';
import { ChevronLeft } from 'lucide-react';
import Link from 'next/link';

export default function ModelsPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => router.back()}
          className="flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white mb-8 transition-colors"
        >
          <ChevronLeft className="w-5 h-5" />
          Back
        </button>

        <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
          Model Documentation
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-300 mb-8">
          Please select a model from the documentation page to view detailed information.
        </p>

        <Link
          href="/docs"
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold transition-colors"
        >
          Back to Documentation
        </Link>
      </div>
    </div>
  );
}
