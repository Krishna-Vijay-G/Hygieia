'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ChevronLeft } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import mermaid from 'mermaid';

interface ModelPageProps {
  params: {
    id: string;
  };
}

const modelMap: Record<string, { filename: string; title: string }> = {
  'heart-prediction': {
    filename: 'Heart Risk Prediction Model.md',
    title: 'Heart Risk Prediction Model',
  },
  'diabetes-prediction': {
    filename: 'Diabetes Risk Prediction Model.md',
    title: 'Diabetes Risk Prediction Model',
  },
  'skin-diagnosis': {
    filename: 'Skin Lesion Diagnosis Model.md',
    title: 'Skin Lesion Diagnostic Model',
  },
  'breast-prediction': {
    filename: 'Breast Cancer Risk Prediction Model.md',
    title: 'Breast Cancer Risk Prediction Model',
  },
  'breast-diagnosis': {
    filename: 'Breast Cancer Tissue Diagnosis Model.md',
    title: 'Breast Cancer Tissue Diagnostic Model',
  },
};

// Mermaid Component
const MermaidDiagram = ({ chart }: { chart: string }) => {
  const [svg, setSvg] = useState<string>('');

  useEffect(() => {
    const renderDiagram = async () => {
      try {
        const { svg } = await mermaid.render(`mermaid-${Date.now()}`, chart);
        setSvg(svg);
      } catch (error) {
        console.error('Mermaid rendering error:', error);
        setSvg('<div class="text-red-500 p-4 border border-red-300 rounded">Error rendering diagram</div>');
      }
    };

    if (chart) {
      renderDiagram();
    }
  }, [chart]);

  return (
    <div
      className="mermaid-diagram my-6 p-4 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
};

export default function ModelPage({ params }: ModelPageProps) {
  const router = useRouter();
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const modelInfo = modelMap[params.id];

  useEffect(() => {
    if (!modelInfo) {
      setError('Model not found');
      setLoading(false);
      return;
    }

    const fetchMarkdown = async () => {
      try {
        const response = await fetch(
          `/model_documentation/${encodeURIComponent(modelInfo.filename)}`
        );
        if (!response.ok) {
          throw new Error('Failed to load documentation');
        }
        const text = await response.text();
        setContent(text);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load documentation');
      } finally {
        setLoading(false);
      }
    };

    fetchMarkdown();
  }, [modelInfo, params.id]);

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

        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              <p className="mt-4 text-slate-600 dark:text-slate-400">Loading documentation...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <h2 className="text-red-800 dark:text-red-200 font-semibold mb-2">Error</h2>
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}

        {!loading && content && (
          <article className="prose prose-slate dark:prose-invert max-w-none">
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-8 [&_pre]:!text-green-400 [&_pre_*]:!text-green-400 [&_code]:!text-slate-900 dark:[&_code]:!text-slate-100">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw]}
                components={{
                  h1: ({ node, ...props }) => (
                    <h1 className="text-4xl font-bold mb-6 text-slate-900 dark:text-white border-b border-slate-200 dark:border-slate-700 pb-3" {...props} />
                  ),
                  h2: ({ node, ...props }) => (
                    <h2 className="text-3xl font-bold mt-10 mb-4 text-slate-900 dark:text-white flex items-center gap-2" {...props} />
                  ),
                  h3: ({ node, ...props }) => (
                    <h3 className="text-2xl font-semibold mt-8 mb-3 text-slate-900 dark:text-white" {...props} />
                  ),
                  h4: ({ node, ...props }) => (
                    <h4 className="text-xl font-semibold mt-6 mb-2 text-slate-900 dark:text-white" {...props} />
                  ),
                  p: ({ node, ...props }) => (
                    <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4 text-lg" {...props} />
                  ),
                  ul: ({ node, ...props }) => (
                    <ul className="list-disc list-inside mb-6 text-slate-700 dark:text-slate-300 space-y-2 ml-4" {...props} />
                  ),
                  ol: ({ node, ...props }) => (
                    <ol className="list-decimal list-inside mb-6 text-slate-700 dark:text-slate-300 space-y-2 ml-4" {...props} />
                  ),
                  li: ({ node, ...props }) => (
                    <li className="leading-relaxed" {...props} />
                  ),
                  blockquote: ({ node, ...props }) => (
                    <blockquote className="border-l-4 border-blue-500 pl-6 py-2 my-6 bg-blue-50 dark:bg-blue-900/20 text-slate-700 dark:text-slate-300 italic" {...props} />
                  ),
                  a: ({ node, ...props }) => (
                    <a className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline decoration-2 underline-offset-2 transition-colors" {...props} />
                  ),
                  table: ({ node, ...props }) => (
                    <div className="overflow-x-auto mb-6 rounded-lg border border-slate-200 dark:border-slate-700">
                      <table className="w-full border-collapse" {...props} />
                    </div>
                  ),
                  th: ({ node, ...props }) => (
                    <th className="border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-700 px-4 py-3 text-left font-semibold text-slate-900 dark:text-white" {...props} />
                  ),
                  td: ({ node, ...props }) => (
                    <td className="border border-slate-200 dark:border-slate-700 px-4 py-3 text-slate-700 dark:text-slate-300" {...props} />
                  ),
                  code: ({ node, ...props }) => (
                    <code className="bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-sm font-mono text-slate-900 dark:text-slate-100" {...props} />
                  ),
                  pre: ({ node, children, ...props }) => {
                    if (!node || !node.children || node.children.length === 0) {
                      return (
                        <pre className="bg-slate-800 dark:bg-slate-900 text-green-400 p-4 rounded-lg overflow-x-auto mb-6 font-mono text-sm border border-slate-600 shadow-lg" {...props}>
                          {children}
                        </pre>
                      );
                    }
                    const child = node.children[0] as any;
                    const isMermaid = child?.tagName === 'code' && child.properties?.className?.includes('language-mermaid');

                    if (isMermaid) {
                      const chart = String(child.children[0]?.value || '');
                      return <MermaidDiagram chart={chart} />;
                    }

                    return (
                      <pre className="bg-slate-800 dark:bg-slate-900 text-green-400 p-4 rounded-lg overflow-x-auto mb-6 font-mono text-sm border border-slate-600 shadow-lg" {...props}>
                        {children}
                      </pre>
                    );
                  },
                  hr: ({ node, ...props }) => (
                    <hr className="border-slate-200 dark:border-slate-700 my-8" {...props} />
                  ),
                  strong: ({ node, ...props }) => (
                    <strong className="font-bold text-slate-900 dark:text-white" {...props} />
                  ),
                  em: ({ node, ...props }) => (
                    <em className="italic text-slate-700 dark:text-slate-300" {...props} />
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          </article>
        )}
      </div>
    </div>
  );
}
