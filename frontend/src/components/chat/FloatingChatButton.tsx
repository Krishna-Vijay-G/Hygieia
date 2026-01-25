'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, X, ExternalLink } from 'lucide-react';
import { useAuthStore } from '@/lib/store';

interface ChatButtonProps {
  analysisId?: string;
}

export function FloatingChatButton({ analysisId }: ChatButtonProps) {
  const { isAuthenticated } = useAuthStore();
  const [isHovered, setIsHovered] = useState(false);

  if (!isAuthenticated) return null;

  const chatUrl = analysisId 
    ? `/chat?analysis_id=${analysisId}` 
    : '/chat';

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.9 }}
            className="absolute bottom-16 right-0 bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700 p-4 w-64"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                <img src="/hygieia_color.png" alt="Dr. Hygieia" className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 dark:text-white text-sm">Dr. Hygieia</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">AI Health Assistant</p>
              </div>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
              {analysisId 
                ? 'Discuss your analysis results with our AI assistant.'
                : 'Get help understanding your health analysis results.'}
            </p>
            <Link
              href={chatUrl}
              className="flex items-center justify-center gap-2 w-full btn-primary btn-sm"
            >
              Start Chat
              <ExternalLink className="w-3.5 h-3.5" />
            </Link>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
      >
        <Link
          href={chatUrl}
          className="flex items-center justify-center w-14 h-14 rounded-full bg-primary-500 text-white shadow-lg shadow-primary-500/30 hover:bg-primary-600 transition-colors"
        >
          <MessageCircle className="w-6 h-6" />
        </Link>
      </motion.div>
    </div>
  );
}
