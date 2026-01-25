'use client';
import { renderToStaticMarkup } from 'react-dom/server';

import { useEffect, useState, useRef } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import html2canvas from 'html2canvas';
import { 
  Heart, 
  Ribbon, 
  Droplets, 
  Scan, 
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  XCircle,
  Shield,
  Clock,
  Download,
  Share2,
  Activity,
  User,
  MessageCircle,
  Sparkles,
  Bot,
  Loader2
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { analysisApi, chatApi } from '@/lib/api';
import { formatDateTime, getAnalysisTypeLabel, getRiskColor, getRiskColorValue, getRiskIconBgColor, getRiskIconColor, truncateHash } from '@/lib/utils';

// Get backend URL for images
const getBackendURL = () => {
  if (typeof window === 'undefined') {
    return 'http://localhost:5000';
  }
  const host = window.location.hostname;
  const protocol = window.location.protocol;
  return `${protocol}//${host}:5000`;
};

const analysisIcons: Record<string, any> = {
  'heart-prediction': Heart,
  'breast-prediction': Ribbon,
  'breast-diagnosis': Ribbon,
  'diabetes-prediction': Droplets,
  'skin-diagnosis': Scan,
};

const analysisColors: Record<string, { bg: string; icon: string }> = {
  'heart-prediction': { bg: 'bg-red-100 dark:bg-red-900/20', icon: 'text-red-600 dark:text-red-400' },
  'breast-prediction': { bg: 'bg-pink-100 dark:bg-pink-900/20', icon: 'text-pink-600 dark:text-pink-400' },
  'breast-diagnosis': { bg: 'bg-fuchsia-100 dark:bg-fuchsia-900/20', icon: 'text-fuchsia-600 dark:text-fuchsia-400' },
  'diabetes-prediction': { bg: 'bg-orange-100 dark:bg-orange-900/20', icon: 'text-orange-600 dark:text-orange-400' },
  'skin-diagnosis': { bg: 'bg-teal-100 dark:bg-teal-900/20', icon: 'text-teal-600 dark:text-teal-400' },
};

// Function to get card background color (CSS color) used in the downloadable image
const getCardBackgroundColor = (analysisType: string) => {
  const colorMap: Record<string, string> = {
    // Use the same gentle background hex colors as the image icon map
    'heart-prediction': '#fee2e2', // gentle red
    'breast-prediction': '#fce7f3', // soft pink
    'breast-diagnosis': '#fae8ff', // light fuchsia/purple
    'diabetes-prediction': '#ffedd5', // light orange
    'skin-diagnosis': '#ccfbf1', // light teal
  };
  return colorMap[analysisType] || '#ffffff'; // fallback to white
};

// Function to get SVG path for each icon type
const getIconPath = (analysisType: string) => {
  const pathMap: Record<string, string> = {
    'heart-prediction': '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>',
    'breast-prediction': '<path d="M12 11a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z"/><path d="m15.47 15.47 3.53 3.53a2 2 0 0 1 0 2.83l-1.41 1.41a2 2 0 0 1-2.83 0L12 20.41l-2.76 2.83a2 2 0 0 1-2.83 0l-1.41-1.41a2 2 0 0 1 0-2.83l3.53-3.53"/><path d="m15 11 4.95 4.95"/><path d="M9 11 4.05 15.95"/>',
    'breast-diagnosis': '<path d="M12 11a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z"/><path d="m15.47 15.47 3.53 3.53a2 2 0 0 1 0 2.83l-1.41 1.41a2 2 0 0 1-2.83 0L12 20.41l-2.76 2.83a2 2 0 0 1-2.83 0l-1.41-1.41a2 2 0 0 1 0-2.83l3.53-3.53"/><path d="m15 11 4.95 4.95"/><path d="M9 11 4.05 15.95"/>',
    'diabetes-prediction': '<path d="M7 16.3c2.2 0 4-1.83 4-4.05 0-1.16-.57-2.26-1.71-3.19S7.29 6.75 7 5.3c-.29 1.45-1.14 2.84-2.29 3.76S3 11.1 3 12.25c0 2.22 1.8 4.05 4 4.05z"/><path d="M12.56 6.6A10.97 10.97 0 0 0 14 3.02c.5 2.5 2 4.9 4 6.5s3 3.5 3 5.5a6.98 6.98 0 0 1-11.91 4.97"/>',
    'skin-diagnosis': '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
  };
  return pathMap[analysisType] || '<path d="M3 7v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2Z"/><path d="M8 5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2H8V5Z"/>';
};

const getRiskIcon = (level: string) => {
  const lower = level?.toLowerCase() || '';
  if (lower.includes('low') || lower.includes('benign') || lower.includes('negative')) {
    return CheckCircle;
  }
  if (lower.includes('high') || lower.includes('malignant') || lower.includes('positive')) {
    return XCircle;
  }
  return AlertTriangle;
};

const getRiskBgColor = (level: string) => {
  const lower = level?.toLowerCase() || '';
  if (lower.includes('low') || lower.includes('benign') || lower.includes('negative')) {
    return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
  }
  if (lower.includes('high') || lower.includes('malignant') || lower.includes('positive')) {
    return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
  }
  return 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800';
};

export default function AnalysisResultPage() {
  const router = useRouter();
  const params = useParams();
  const analysisId = params.id as string;
  const { isAuthenticated, isLoading: authLoading } = useAuthStore();
  const contentRef = useRef<HTMLDivElement>(null);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [canvasData, setCanvasData] = useState<{ dataUrl: string; width: number; height: number; size: number } | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  const handleDownload = async () => {
    if (!analysis) return;

    // Create a temporary container for the image - make it taller for more content
    const tempContainer = document.createElement('div');
    tempContainer.style.position = 'absolute';
    tempContainer.style.left = '-9999px';
    tempContainer.style.top = '-9999px';
    tempContainer.style.width = '1200px'; // 16:9 aspect ratio (1200x675)
    tempContainer.style.height = '800px'; // Increased height for more content
    tempContainer.style.background = '#ffffff'; // White background instead of grey gradient
    tempContainer.style.padding = '40px';
    tempContainer.style.boxSizing = 'border-box';
    tempContainer.style.fontFamily = 'system-ui, -apple-system, sans-serif';
    tempContainer.style.color = '#1e293b'; // Slate-800
    tempContainer.style.overflow = 'hidden';

    // Header section
    const header = document.createElement('div');
    header.style.position = 'relative';
    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.justifyContent = 'space-between';
    header.style.marginBottom = '20px';
    
    // Left side content
    const leftContent = document.createElement('div');
    leftContent.style.display = 'flex';
    leftContent.style.alignItems = 'center';
    
    const iconDiv = document.createElement('div');
    iconDiv.style.width = '50px';
    iconDiv.style.height = '50px';
    iconDiv.style.borderRadius = '12px';
    iconDiv.style.display = 'flex';
    iconDiv.style.alignItems = 'center';
    iconDiv.style.justifyContent = 'center';
    iconDiv.style.marginRight = '15px';
    
    // Get analysis-specific icon and colors (matching lucide-react icons)
    const analysisIconMap: Record<string, { icon: any; bg: string; color: string }> = {
      'heart-prediction': {icon: Heart, bg: '#fee2e2', color: '#dc2626' },
      'breast-prediction': {icon: Ribbon, bg: '#fce7f3', color: '#db2777' },
      'breast-diagnosis': {icon: Ribbon, bg: '#fae8ff', color: '#c026d3' },
      'diabetes-prediction': {icon: Droplets, bg: '#ffedd5', color: '#ea580c' },
      'skin-diagnosis': {icon: Scan, bg: '#ccfbf1', color: '#0d9488' },
    };
    
    const analysisConfig = analysisIconMap[analysis.analysis_type] || { 
      icon: Activity,
      bg: '#eff6ff', 
      color: '#2563eb' 
    };
    iconDiv.style.background = analysisConfig.bg;
    
    const icon = document.createElement('div');
    icon.style.width = '24px';
    icon.style.height = '24px';
    icon.style.color = analysisConfig.color;

    // Create SVG element directly for better html2canvas compatibility
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '24');
    svg.setAttribute('height', '24');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', analysisConfig.color);
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');

    // Get the path data from the Lucide icon component
    const IconComponent = analysisConfig.icon;
    const tempDiv = document.createElement('div');
    const markup = renderToStaticMarkup(<IconComponent />);
    tempDiv.innerHTML = markup;
    const pathElements = tempDiv.querySelectorAll('path');
    
    pathElements.forEach(path => {
      const newPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      newPath.setAttribute('d', path.getAttribute('d') || '');
      svg.appendChild(newPath);
    });

    icon.appendChild(svg);
    iconDiv.appendChild(icon);
    leftContent.appendChild(iconDiv);

    const titleDiv = document.createElement('div');
    const title = document.createElement('h1');
    title.style.fontSize = '24px';
    title.style.fontWeight = 'bold';
    title.style.margin = '0 0 2px 0';
    title.style.color = '#1e293b';
    title.textContent = `${getAnalysisTypeLabel(analysis.analysis_type)} Report`;
    titleDiv.appendChild(title);

    // Meta row: generated time + analysed by (if available)
    const metaRow = document.createElement('div');
    metaRow.style.display = 'flex';
    metaRow.style.alignItems = 'center';
    metaRow.style.gap = '10px';

    const date = document.createElement('p');
    date.style.fontSize = '12px';
    date.style.opacity = '0.7';
    date.style.margin = '0';
    date.style.color = '#64748b';
    date.textContent = `Generated on ${formatDateTime(analysis.created_at)}`;
    metaRow.appendChild(date);

    if (analysis.user) {
      const analyzedBy = document.createElement('p');
      analyzedBy.style.fontSize = '12px';
      analyzedBy.style.opacity = '0.7';
      analyzedBy.style.margin = '0';
      analyzedBy.style.color = '#64748b';
      analyzedBy.textContent = `Analysed by ${analysis.user.full_name} (${analysis.user.username})`;
      metaRow.appendChild(analyzedBy);
    }

    titleDiv.appendChild(metaRow);
    leftContent.appendChild(titleDiv);
    header.appendChild(leftContent);
    
    // Right side - Hygieia branding (text top, logo bottom-right)
    const rightContent = document.createElement('div');
    rightContent.style.display = 'flex';
    rightContent.style.flexDirection = 'column';
    rightContent.style.justifyContent = 'space-between';
    rightContent.style.alignItems = 'flex-end';
    rightContent.style.gap = '12px';
    // Make the right column the same height as the icon so text sits top and icon sits bottom
    rightContent.style.height = '80px';

    const hygieiaTextDiv = document.createElement('div');
    hygieiaTextDiv.style.display = 'flex';
    hygieiaTextDiv.style.flexDirection = 'column';
    hygieiaTextDiv.style.justifyContent = 'flex-start';
    hygieiaTextDiv.style.alignItems = 'flex-end';
    hygieiaTextDiv.style.paddingTop = '0px';

    const hygieiTitle = document.createElement('h1');
    hygieiTitle.style.fontSize = '24px';
    hygieiTitle.style.fontWeight = 'bold';
    hygieiTitle.style.margin = '0';
    hygieiTitle.style.color = '#1e293b';
    hygieiTitle.textContent = 'Hygieia';
    hygieiaTextDiv.appendChild(hygieiTitle);

    // small subtitle or spacer if needed (keeps vertical rhythm)
    const hygieiSub = document.createElement('p');
    hygieiSub.style.margin = '0';
    hygieiSub.style.fontSize = '12px';
    hygieiSub.style.opacity = '0.0';
    hygieiSub.textContent = '';
    hygieiaTextDiv.appendChild(hygieiSub);

    // text is right-aligned at the top of the header; no header icon (footer will hold the logo)
    hygieiaTextDiv.style.marginRight = '0';
    rightContent.appendChild(hygieiaTextDiv);

    const hygieiaIconDiv = document.createElement('div');
    hygieiaIconDiv.style.width = '120px';
    hygieiaIconDiv.style.height = '120px';
    hygieiaIconDiv.style.borderRadius = '12px';
    hygieiaIconDiv.style.display = 'flex';
    hygieiaIconDiv.style.alignItems = 'center';
    hygieiaIconDiv.style.justifyContent = 'center';
    hygieiaIconDiv.style.marginLeft = '0';

    const hygieiaIconImg = document.createElement('img');
    hygieiaIconImg.src = '/Hero.svg';
    hygieiaIconImg.style.width = '120px';
    hygieiaIconImg.style.height = '120px';
    hygieiaIconImg.style.objectFit = 'contain';
    hygieiaIconImg.style.display = 'block';
    hygieiaIconImg.style.alignSelf = 'flex-start';
    hygieiaIconDiv.appendChild(hygieiaIconImg);

    // DO NOT append the icon to the header; we'll add it to the footer area below

    header.appendChild(rightContent);

    tempContainer.appendChild(header);

    // Main content grid - adjust for more content
    const contentGrid = document.createElement('div');
    contentGrid.style.display = 'grid';
    contentGrid.style.gridTemplateColumns = '1fr 320px'; // Wider right column
    contentGrid.style.gap = '20px';
    contentGrid.style.height = 'calc(100% - 100px)';

    // Left column - Main content
    const leftColumn = document.createElement('div');
    leftColumn.style.display = 'flex';
    leftColumn.style.flexDirection = 'column';
    leftColumn.style.gap = '15px';

    // Risk Assessment Card
    const riskCard = document.createElement('div');
    riskCard.style.background = getCardBackgroundColor(analysis.analysis_type);
    riskCard.style.borderRadius = '12px';
    riskCard.style.padding = '20px';
    riskCard.style.backdropFilter = 'blur(10px)';
    riskCard.style.border = '1px solid rgba(148, 163, 184, 0.1)';
    riskCard.style.boxShadow = '0 8px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';

    const riskTitle = document.createElement('h2');
    riskTitle.style.fontSize = '16px';
    riskTitle.style.fontWeight = '600';
    riskTitle.style.margin = '0 0 15px 0';
    riskTitle.style.textTransform = 'uppercase';
    riskTitle.style.letterSpacing = '0.5px';
    riskTitle.style.color = '#64748b';
    riskTitle.textContent = 'Assessment Result';
    riskCard.appendChild(riskTitle);

    if (analysis.result_data?.condition_name) {
      const condition = document.createElement('h3');
      condition.style.fontSize = '22px';
      condition.style.fontWeight = 'bold';
      condition.style.margin = '0 0 8px 0';
      condition.style.color = '#1e293b';
      condition.textContent = analysis.result_data.condition_name;
      riskCard.appendChild(condition);
    }

    const riskLevel = document.createElement('p');
    riskLevel.style.fontSize = '18px';
    riskLevel.style.fontWeight = '600';
    riskLevel.style.margin = '0 0 15px 0';
    riskLevel.style.color = getRiskColorValue(analysis.risk_level || '');
    riskLevel.textContent = analysis.risk_level || 'N/A';
    riskCard.appendChild(riskLevel);

    if (analysis.confidence) {
      const confidenceDiv = document.createElement('div');
      confidenceDiv.style.marginTop = '15px';
      
      const confidenceLabel = document.createElement('div');
      confidenceLabel.style.fontSize = '12px';
      confidenceLabel.style.opacity = '0.8';
      confidenceLabel.style.marginBottom = '5px';
      confidenceLabel.style.color = '#64748b';
      confidenceLabel.textContent = 'Confidence Score';
      confidenceDiv.appendChild(confidenceLabel);

      const confidenceValue = document.createElement('div');
      confidenceValue.style.fontSize = '14px';
      confidenceValue.style.fontWeight = '600';
      confidenceValue.style.color = '#1e293b';
      confidenceValue.textContent = `${(analysis.confidence * 100).toFixed(1)}%`;
      confidenceDiv.appendChild(confidenceValue);

      riskCard.appendChild(confidenceDiv);
    }

    leftColumn.appendChild(riskCard);

    // Analysis Details Card - Moved to left column between assessment and AI summary
    const detailsCard = document.createElement('div');
    detailsCard.style.background = getCardBackgroundColor(analysis.analysis_type);
    detailsCard.style.borderRadius = '12px';
    detailsCard.style.padding = '15px';
    detailsCard.style.backdropFilter = 'blur(10px)';
    detailsCard.style.border = '1px solid rgba(148, 163, 184, 0.1)';
    detailsCard.style.boxShadow = '0 8px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';

    const detailsTitle = document.createElement('h4');
    detailsTitle.style.fontSize = '12px';
    detailsTitle.style.fontWeight = '600';
    detailsTitle.style.margin = '0 0 10px 0';
    detailsTitle.style.color = '#64748b';
    detailsTitle.style.textTransform = 'uppercase';
    detailsTitle.style.letterSpacing = '0.5px';
    detailsTitle.textContent = 'Analysis Details';
    detailsCard.appendChild(detailsTitle);

    const detailsList = document.createElement('div');
    detailsList.style.display = 'flex';
    detailsList.style.flexDirection = 'column';
    detailsList.style.gap = '6px';

    // Analysis ID
    const analysisIdDiv = document.createElement('div');
    analysisIdDiv.style.display = 'flex';
    analysisIdDiv.style.justifyContent = 'space-between';
    analysisIdDiv.style.alignItems = 'center';

    const idLabel = document.createElement('span');
    idLabel.style.fontSize = '11px';
    idLabel.style.color = '#64748b';
    idLabel.textContent = 'Analysis ID:';
    analysisIdDiv.appendChild(idLabel);

    const idValue = document.createElement('span');
    idValue.style.fontSize = '10px';
    idValue.style.fontFamily = 'monospace';
    idValue.style.color = '#1e293b';
    idValue.style.background = 'rgba(241, 245, 249, 0.8)';
    idValue.style.padding = '2px 4px';
    idValue.style.borderRadius = '3px';
    idValue.textContent = analysis.id;
    analysisIdDiv.appendChild(idValue);

    detailsList.appendChild(analysisIdDiv);

    // Analysis Method
    if (analysis.result_data?.method) {
      const methodDiv = document.createElement('div');
      methodDiv.style.display = 'flex';
      methodDiv.style.justifyContent = 'space-between';
      methodDiv.style.alignItems = 'center';

      const methodLabel = document.createElement('span');
      methodLabel.style.fontSize = '11px';
      methodLabel.style.color = '#64748b';
      methodLabel.textContent = 'Method:';
      methodDiv.appendChild(methodLabel);

      const methodValue = document.createElement('span');
      methodValue.style.fontSize = '10px';
      methodValue.style.color = '#1e293b';
      methodValue.textContent = analysis.result_data.method;
      methodDiv.appendChild(methodValue);

      detailsList.appendChild(methodDiv);
    }

    // Model Name - Show full name
    if (analysis.model_name) {
      const modelDiv = document.createElement('div');
      modelDiv.style.display = 'flex';
      modelDiv.style.justifyContent = 'space-between';
      modelDiv.style.alignItems = 'center';

      const modelLabel = document.createElement('span');
      modelLabel.style.fontSize = '11px';
      modelLabel.style.color = '#64748b';
      modelLabel.textContent = 'Model:';
      modelDiv.appendChild(modelLabel);

      const modelValue = document.createElement('span');
      modelValue.style.fontSize = '10px';
      modelValue.style.color = '#1e293b';
      modelValue.textContent = analysis.model_name; // Show full model name
      modelDiv.appendChild(modelValue);

      detailsList.appendChild(modelDiv);
    }

    // Blockchain Hash - Show full hash
    if (analysis.blockchain_hash) {
      const hashDiv = document.createElement('div');
      hashDiv.style.display = 'flex';
      hashDiv.style.justifyContent = 'space-between';
      hashDiv.style.alignItems = 'flex-start';
      hashDiv.style.gap = '8px';

      const hashLabel = document.createElement('span');
      hashLabel.style.fontSize = '11px';
      hashLabel.style.color = '#64748b';
      hashLabel.textContent = 'Block Hash:';
      hashDiv.appendChild(hashLabel);

      const hashValue = document.createElement('span');
      hashValue.style.fontSize = '9px';
      hashValue.style.fontFamily = 'monospace';
      hashValue.style.color = '#1e293b';
      hashValue.style.background = 'rgba(241, 245, 249, 0.8)';
      hashValue.style.padding = '2px 4px';
      hashValue.style.borderRadius = '3px';
      hashValue.style.wordBreak = 'break-all';
      hashValue.style.maxWidth = '60%'; // Limit width to prevent overflow
      hashValue.textContent = analysis.blockchain_hash; // Show full hash
      hashDiv.appendChild(hashValue);

      detailsList.appendChild(hashDiv);
    }

    // (Moved 'Analysed by' to the header meta row — removed from details list)

    detailsCard.appendChild(detailsList);
    leftColumn.appendChild(detailsCard);

    // AI Summary Card - Add below assessment result
    try {
      const summaryResponse = await chatApi.getAnalysisSummary(analysisId);
      if (summaryResponse.summary) {
        const summaryCard = document.createElement('div');
        summaryCard.style.background = getCardBackgroundColor(analysis.analysis_type);
        summaryCard.style.borderRadius = '12px';
        summaryCard.style.padding = '20px';
        summaryCard.style.backdropFilter = 'blur(10px)';
        summaryCard.style.border = '1px solid rgba(148, 163, 184, 0.1)';
        summaryCard.style.boxShadow = '0 8px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';
        // Removed maxHeight restriction to allow full content display

        const summaryTitle = document.createElement('h3');
        summaryTitle.style.fontSize = '14px';
        summaryTitle.style.fontWeight = '600';
        summaryTitle.style.margin = '0 0 10px 0';
        summaryTitle.style.color = '#64748b';
        summaryTitle.style.textTransform = 'uppercase';
        summaryTitle.style.letterSpacing = '0.5px';
        summaryTitle.textContent = "Hygieia's Summary";
        summaryCard.appendChild(summaryTitle);

        const summaryContent = document.createElement('div');
        summaryContent.style.fontSize = '12px';
        summaryContent.style.lineHeight = '1.4';
        summaryContent.style.color = '#475569';
        // Removed maxHeight and overflow restrictions to show all content
        
        // Simple text extraction from markdown (basic implementation)
        const plainText = summaryResponse.summary
          .replace(/#{1,6}\s*/g, '') // Remove headers
          .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
          .replace(/\*(.*?)\*/g, '$1') // Remove italic
          .replace(/```[\s\S]*?```/g, '') // Remove code blocks
          .replace(/`(.*?)`/g, '$1') // Remove inline code
          .replace(/^\s*[-*+]\s+/gm, '') // Remove list markers
          .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered list markers
          .replace(/\n\s*\n/g, '\n') // Remove extra newlines
          .trim();
        
        summaryContent.textContent = plainText; // Show full content without truncation
        summaryCard.appendChild(summaryContent);

        leftColumn.appendChild(summaryCard);
      } else {
        // Add a placeholder if no summary is available
        const summaryCard = document.createElement('div');
        summaryCard.style.background = getCardBackgroundColor(analysis.analysis_type);
        summaryCard.style.borderRadius = '12px';
        summaryCard.style.padding = '20px';
        summaryCard.style.backdropFilter = 'blur(10px)';
        summaryCard.style.border = '1px solid rgba(148, 163, 184, 0.1)';
        summaryCard.style.boxShadow = '0 8px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';

        const summaryTitle = document.createElement('h3');
        summaryTitle.style.fontSize = '14px';
        summaryTitle.style.fontWeight = '600';
        summaryTitle.style.margin = '0 0 10px 0';
        summaryTitle.style.color = '#64748b';
        summaryTitle.style.textTransform = 'uppercase';
        summaryTitle.style.letterSpacing = '0.5px';
        summaryTitle.textContent = "Hygieia's Summary";
        summaryCard.appendChild(summaryTitle);

        const summaryContent = document.createElement('div');
        summaryContent.style.fontSize = '12px';
        summaryContent.style.lineHeight = '1.4';
        summaryContent.style.color = '#94a3b8';
        summaryContent.style.fontStyle = 'italic';
        summaryContent.textContent = 'AI summary not available for this analysis.';
        summaryCard.appendChild(summaryContent);

        leftColumn.appendChild(summaryCard);
      }
    } catch (error) {
      console.log('Could not load AI summary for download');
      // Add a placeholder if API call fails
      const summaryCard = document.createElement('div');
      summaryCard.style.background = getCardBackgroundColor(analysis.analysis_type);
      summaryCard.style.borderRadius = '12px';
      summaryCard.style.padding = '20px';
      summaryCard.style.backdropFilter = 'blur(10px)';
      summaryCard.style.border = '1px solid rgba(148, 163, 184, 0.1)';
      summaryCard.style.boxShadow = '0 8px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';

      const summaryTitle = document.createElement('h3');
      summaryTitle.style.fontSize = '14px';
      summaryTitle.style.fontWeight = '600';
      summaryTitle.style.margin = '0 0 10px 0';
      summaryTitle.style.color = '#64748b';
      summaryTitle.style.textTransform = 'uppercase';
      summaryTitle.style.letterSpacing = '0.5px';
      summaryTitle.textContent = "Hygieia's Summary";
      summaryCard.appendChild(summaryTitle);

      const summaryContent = document.createElement('div');
      summaryContent.style.fontSize = '12px';
      summaryContent.style.lineHeight = '1.4';
      summaryContent.style.color = '#94a3b8';
      summaryContent.style.fontStyle = 'italic';
      summaryContent.textContent = 'AI summary not available for this analysis.';
      summaryCard.appendChild(summaryContent);

      leftColumn.appendChild(summaryCard);
    }

    contentGrid.appendChild(leftColumn);

    // Right column - Additional info
    const rightColumn = document.createElement('div');
    rightColumn.style.display = 'flex';
    rightColumn.style.flexDirection = 'column';
    rightColumn.style.gap = '15px';

    // Input Parameters Card - Moved to right column above analysis details
    if (analysis.input_data && Object.keys(analysis.input_data).length > 0) {
      const paramsCard = document.createElement('div');
      paramsCard.style.background = getCardBackgroundColor(analysis.analysis_type);
      paramsCard.style.borderRadius = '12px';
      paramsCard.style.padding = '20px';
      paramsCard.style.backdropFilter = 'blur(10px)';
      paramsCard.style.border = '1px solid rgba(148, 163, 184, 0.1)';
      paramsCard.style.boxShadow = '0 8px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';

      const paramsTitle = document.createElement('h3');
      paramsTitle.style.fontSize = '14px';
      paramsTitle.style.fontWeight = '600';
      paramsTitle.style.margin = '0 0 15px 0';
      paramsTitle.style.color = '#64748b';
      paramsTitle.style.textTransform = 'uppercase';
      paramsTitle.style.letterSpacing = '0.5px';
      paramsTitle.textContent = 'Input Parameters';
      paramsCard.appendChild(paramsTitle);

      const paramsList = document.createElement('div');
      paramsList.style.display = 'grid';
      paramsList.style.gridTemplateColumns = '1fr auto';
      paramsList.style.gap = '8px 12px';
      paramsList.style.alignItems = 'center';

      Object.entries(analysis.input_data).forEach(([key, value]) => {
        // For breast prediction, only show actual_* fields
        if (analysis.analysis_type === 'breast-prediction' && !key.startsWith('actual_')) {
          return;
        }

        // Map field names for breast prediction
        let displayKey = key.replace(/_/g, ' ');
        if (analysis.analysis_type === 'breast-prediction') {
          const fieldMappings: Record<string, string> = {
            'actual_age': 'Age',
            'actual_bmi': 'BMI',
            'actual_density': 'Breast Density',
            'actual_race': 'Race/Ethnicity',
            'actual_age_menarche': 'Age at Menarche',
            'actual_age_first_birth': 'Age at First Birth',
            'actual_brstproc': 'Previous Biopsy',
            'actual_hrt': 'HRT Use',
            'actual_family_hx': 'Family History',
            'actual_menopaus': 'Menopausal Status',
          };
          displayKey = fieldMappings[key] || key.replace(/actual_/g, '');
        } else if (analysis.analysis_type === 'breast-diagnosis') {
          // Convert to title case for breast diagnosis
          displayKey = displayKey.replace(/\b\w/g, l => l.toUpperCase());
        } else if (analysis.analysis_type === 'diabetes-prediction') {
          // Convert to title case for diabetes prediction
          displayKey = displayKey.replace(/\b\w/g, l => l.toUpperCase());
        }

        // Format parameter value; special-case Heart Prediction binary/gender fields
        let paramValue = '';
        if (analysis.analysis_type === 'heart-prediction') {
          const normalizedKey = String(key).toLowerCase();
          const normalizedVal = String(value).toLowerCase();

          if (normalizedKey === 'gender' || normalizedKey === 'sex') {
            if (normalizedVal === '1' || normalizedVal === 'male' || normalizedVal === 'm') {
              paramValue = 'Male';
            } else if (normalizedVal === '0' || normalizedVal === 'female' || normalizedVal === 'f') {
              paramValue = 'Female';
            } else {
              paramValue = String(value);
            }
          } else if (normalizedKey !== 'age' && (normalizedVal === '0' || normalizedVal === '1' || value === 0 || value === 1 || typeof value === 'boolean')) {
            paramValue = (value === 1 || value === '1' || value === true || normalizedVal === '1') ? 'Yes' : 'No';
          } else {
            paramValue = typeof value === 'boolean' 
              ? (value ? 'Yes' : 'No')
              : typeof value === 'number'
              ? value.toFixed(value % 1 === 0 ? 0 : 2)
              : String(value);
          }
        } else {
          paramValue = typeof value === 'boolean' 
            ? (value ? 'Yes' : 'No')
            : typeof value === 'number'
            ? value.toFixed(value % 1 === 0 ? 0 : 2)
            : String(value);
        }

        // Label column
        const paramLabel = document.createElement('span');
        paramLabel.style.fontSize = '11px';
        paramLabel.style.color = '#64748b';
        paramLabel.style.fontWeight = '500';
        paramLabel.textContent = `${displayKey}:`;
        paramsList.appendChild(paramLabel);

        // Value column
        const paramValueSpan = document.createElement('span');
        paramValueSpan.style.fontSize = '11px';
        paramValueSpan.style.color = '#1e293b';
        paramValueSpan.style.fontWeight = '600';
        paramValueSpan.style.textAlign = 'left';
        paramValueSpan.textContent = paramValue;
        paramsList.appendChild(paramValueSpan);
      });

      paramsCard.appendChild(paramsList);
      rightColumn.appendChild(paramsCard);
    }

    // Image for skin diagnosis
    if (analysis.analysis_type === 'skin-diagnosis' && analysis.image_path) {
      const imageCard = document.createElement('div');
      imageCard.style.background = getCardBackgroundColor(analysis.analysis_type);
      imageCard.style.borderRadius = '12px';
      imageCard.style.padding = '15px';
      imageCard.style.backdropFilter = 'blur(10px)';
      imageCard.style.border = '1px solid rgba(148, 163, 184, 0.1)';
      imageCard.style.boxShadow = '0 8px 25px -5px rgba(0, 0, 0, 0.15), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';

      const imageTitle = document.createElement('h4');
      imageTitle.style.fontSize = '12px';
      imageTitle.style.fontWeight = '600';
      imageTitle.style.margin = '0 0 10px 0';
      imageTitle.style.color = '#64748b';
      imageTitle.style.textTransform = 'uppercase';
      imageTitle.style.letterSpacing = '0.5px';
      imageTitle.textContent = 'Analyzed Image';
      imageCard.appendChild(imageTitle);

      const img = document.createElement('img');
      img.src = `${getBackendURL()}/uploads/${analysis.image_path}`;
      img.style.width = '100%';
      img.style.height = 'auto';
      img.style.borderRadius = '6px';
      img.style.objectFit = 'contain';
      img.crossOrigin = 'anonymous';
      
      imageCard.appendChild(img);
      rightColumn.appendChild(imageCard);
    }

    contentGrid.appendChild(rightColumn);
    tempContainer.appendChild(contentGrid);

    // Footer
    const footer = document.createElement('div');
    footer.style.position = 'absolute';
    footer.style.bottom = '20px';
    footer.style.left = '40px';
    footer.style.right = '40px';
    footer.style.textAlign = 'center';
    footer.style.fontSize = '10px';
    footer.style.opacity = '0.6';
    footer.style.color = '#64748b';
    
    const footerText = document.createElement('div');
    footerText.textContent = 'Hygieia AI Medical Assistant • Confidential Medical Report';
    footer.appendChild(footerText);

    tempContainer.appendChild(footer);

    // Place Hygieia logo in the footer area (bottom-right corner of the image)
    hygieiaIconDiv.style.position = 'absolute';
    hygieiaIconDiv.style.right = '40px';
    hygieiaIconDiv.style.bottom = '20px';
    hygieiaIconDiv.style.zIndex = '10';
    tempContainer.appendChild(hygieiaIconDiv);

    // Add to DOM temporarily
    document.body.appendChild(tempContainer);

    try {
      const canvas = await html2canvas(tempContainer, {
        scale: 2,
        useCORS: true,
        allowTaint: false,
        backgroundColor: '#ffffff',
        height: 800, // Match the container height
      });
      
      const dataUrl = canvas.toDataURL('image/png');
      const width = canvas.width;
      const height = canvas.height;
      const size = Math.round((dataUrl.length * 3) / 4);
      
      setCanvasData({ dataUrl, width, height, size });
      setShowDownloadModal(true);
    } catch (error) {
      console.error('Error generating canvas:', error);
    } finally {
      // Remove temporary container
      document.body.removeChild(tempContainer);
    }
  };

  const handleActualDownload = () => {
    if (!canvasData) return;

    const link = document.createElement('a');
    link.download = `analysis-result-${analysisId}.png`;
    link.href = canvasData.dataUrl;
    link.click();
    setShowDownloadModal(false);
  };

  const { data, isLoading, error } = useQuery({
    queryKey: ['analysis', analysisId],
    queryFn: () => analysisApi.getAnalysis(analysisId),
    enabled: isAuthenticated && !!analysisId,
    retry: false, // Don't retry on 403/404 errors
  });

  if (authLoading || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  // Check for access denied error (403)
  const isAccessDenied = error && (error as any)?.response?.status === 403;
  
  if (isAccessDenied) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white dark:bg-slate-800 rounded-lg shadow-lg p-8 text-center">
          <AlertCircle className="w-16 h-16 text-red-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Access Denied</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            You can only view your own analysis results. This analysis belongs to another user.
          </p>
          <Link
            href="/history"
            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            View My Analysis History
          </Link>
        </div>
      </div>
    );
  }

  if (error || !data?.analysis) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white dark:bg-slate-800 rounded-lg shadow-lg p-8 text-center">
          <AlertCircle className="w-16 h-16 text-red-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
            Analysis Not Found
          </h2>
          <p className="text-slate-600 dark:text-slate-400 mb-4">
            The analysis you're looking for doesn't exist or you don't have access.
          </p>
          <Link href="/dashboard" className="btn btn-primary px-6">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  const analysis = data.analysis;
  const Icon = analysisIcons[analysis.analysis_type] || Activity;
  const colors = analysisColors[analysis.analysis_type] || { bg: 'bg-slate-100', icon: 'text-slate-600' };
  const RiskIcon = getRiskIcon(analysis.risk_level || '');
  const riskBgColor = getRiskBgColor(analysis.risk_level || '');
  const riskIconBgColor = getRiskIconBgColor(analysis.risk_level || '');
  const riskIconColor = getRiskIconColor(analysis.risk_level || '');

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div ref={contentRef} className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Back Link */}
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Dashboard
        </Link>

        <div ref={contentRef}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-8 mb-6"
        >
          <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
            <div className={`p-4 rounded-2xl ${colors.bg}`}>
              <Icon className={`w-10 h-10 ${colors.icon}`} />
            </div>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
                {getAnalysisTypeLabel(analysis.analysis_type)} Result
              </h1>
              <div className="flex flex-wrap items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  {formatDateTime(analysis.created_at)}
                </span>
                {analysis.blockchain_status === 'verified' && (
                  <span className="flex items-center gap-1">
                    <Shield className="w-4 h-4 text-green-500" />
                    Verified on blockchain
                  </span>
                )}
                {analysis.blockchain_status === 'failed' && (
                  <span className="flex items-center gap-1">
                    <Shield className="w-4 h-4 text-red-500" />
                    Verification Failed
                  </span>
                )}
                {analysis.blockchain_status === 'not_secured' && analysis.blockchain_hash && (
                  <span className="flex items-center gap-1">
                    <Shield className="w-4 h-4 text-amber-500" />
                    Security Pending
                  </span>
                )}
                {analysis.user && (
                  <span className="flex items-center gap-1">
                    <User className="w-4 h-4" />
                    Analysed by {analysis.user.full_name} ({analysis.user.username})
                  </span>
                )}
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={handleDownload} className="btn btn-outline p-2">
                <Download className="w-5 h-5" />
              </button>
              <button className="btn btn-outline p-2">
                <Share2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </motion.div>

        {/* Risk Assessment */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className={`card p-8 mb-6 border-2 ${riskBgColor}`}
        >
          <div className="flex flex-col md:flex-row items-center gap-6 text-center md:text-left">
            <div className={`p-4 rounded-full ${riskIconBgColor}`}>
              <RiskIcon className={`w-12 h-12 ${riskIconColor}`} />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                Assessment Result
              </p>
              {analysis.analysis_type === 'skin-diagnosis' && analysis.result_data?.condition_name && (
                <div className="mb-3">
                  <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-1">
                    {analysis.result_data.condition_name}
                  </h2>
                  <p className={`text-xl font-semibold ${getRiskColor(analysis.risk_level || '')}`}>
                    {analysis.risk_level || 'N/A'}
                  </p>
                </div>
              )}
              {(analysis.analysis_type === 'breast-prediction' || analysis.analysis_type === 'breast-diagnosis') && analysis.result_data?.condition_name && (
                <div className="mb-3">
                  <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-1">
                    {analysis.result_data.condition_name}
                  </h2>
                  <p className={`text-xl font-semibold ${getRiskColor(analysis.risk_level || '')}`}>
                    {analysis.risk_level || 'N/A'}
                  </p>
                </div>
              )}
              {analysis.analysis_type !== 'skin-diagnosis' && analysis.analysis_type !== 'breast-prediction' && analysis.analysis_type !== 'breast-diagnosis' && (
                <h2 className={`text-3xl font-bold ${analysis.risk_level ? getRiskColor(analysis.risk_level) : 'text-purple-600 dark:text-purple-400'}`}>
                  {analysis.risk_level || 'N/A'}
                </h2>
              )}
              {analysis.confidence && (
                <div className="mt-4">
                  <div className="flex items-center justify-center md:justify-start gap-2 mb-1">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Confidence Score
                    </span>
                    <span className="font-semibold text-slate-900 dark:text-white">
                      {(analysis.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full md:w-64 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary-500 rounded-full transition-all duration-500"
                      style={{ width: `${analysis.confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Input Data */}
        {analysis.input_data && Object.keys(analysis.input_data).length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-8 mb-6"
          >
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Input Parameters
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {Object.entries(analysis.input_data).map(([key, value]) => {
                // For breast prediction, ONLY show user-friendly actual_* fields
                if (analysis.analysis_type === 'breast-prediction') {
                  // Only show fields that start with "actual_"
                  if (!key.startsWith('actual_')) {
                    return null;
                  }
                  
                  // Map actual field names to friendly display names
                  const fieldMappings: Record<string, string> = {
                    'actual_age': 'Age',
                    'actual_bmi': 'BMI',
                    'actual_density': 'Breast Density',
                    'actual_race': 'Race/Ethnicity',
                    'actual_age_menarche': 'Age at Menarche',
                    'actual_age_first_birth': 'Age at First Birth',
                    'actual_brstproc': 'Previous Biopsy',
                    'actual_hrt': 'HRT Use',
                    'actual_family_hx': 'Family History',
                    'actual_menopaus': 'Menopausal Status',
                  };
                  
                  const displayKey = fieldMappings[key] || key.replace(/actual_/g, '').replace(/_/g, ' ');
                  const displayValue = value;
                  
                  return (
                    <div
                      key={key}
                      className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg"
                    >
                      <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                        {displayKey}
                      </p>
                      <p className="font-medium text-slate-900 dark:text-white">
                        {typeof displayValue === 'boolean' 
                          ? (displayValue ? 'Yes' : 'No')
                          : typeof displayValue === 'number'
                          ? displayValue.toFixed(displayValue % 1 === 0 ? 0 : 2)
                          : String(displayValue)
                        }
                      </p>
                    </div>
                  );
                }
                
                // For other analysis types, show all fields normally
                let displayValue = value;
                let displayKey = key.replace(/_/g, ' ');
                
                // Special handling for Heart Prediction numeric values (case-insensitive)
                if (analysis.analysis_type === 'heart-prediction') {
                  const normalizedKey = String(key).toLowerCase();
                  const normalizedVal = String(value).toLowerCase();

                  // Map gender/sex fields to Male/Female
                  if (normalizedKey === 'gender' || normalizedKey === 'sex') {
                    if (normalizedVal === '1' || normalizedVal === 'male' || normalizedVal === 'm') {
                      displayValue = 'Male';
                    } else if (normalizedVal === '0' || normalizedVal === 'female' || normalizedVal === 'f') {
                      displayValue = 'Female';
                    } else {
                      // fallback to string representation
                      displayValue = String(value);
                    }

                  // For other binary fields (excluding age), map 0/1/boolean to Yes/No
                  } else if (normalizedKey !== 'age' && (normalizedVal === '0' || normalizedVal === '1' || value === 0 || value === 1 || typeof value === 'boolean')) {
                    displayValue = (value === 1 || value === '1' || value === true || normalizedVal === '1') ? 'Yes' : 'No';
                  }
                }
                
                return (
                  <div
                    key={key}
                    className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg"
                  >
                    <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                      {displayKey}
                    </p>
                    <p className="font-medium text-slate-900 dark:text-white">
                      {typeof displayValue === 'boolean' 
                        ? (displayValue ? 'Yes' : 'No')
                        : typeof displayValue === 'number'
                        ? displayValue.toFixed(displayValue % 1 === 0 ? 0 : 2)
                        : String(displayValue)
                      }
                    </p>
                  </div>
                );
              }).filter(Boolean)}
            </div>
          </motion.div>
        )}

        {/* Image Display for Skin Diagnosis */}
        {analysis.analysis_type === 'skin-diagnosis' && analysis.image_path && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-8 mb-6"
          >
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Analyzed Image
            </h3>
            <div className="flex justify-center">
              <img
                src={`${getBackendURL()}/uploads/${analysis.image_path}`}
                alt="Analyzed skin lesion"
                className="max-w-md max-h-96 rounded-lg border-2 border-slate-200 dark:border-slate-700"
                onError={(e) => {
                  console.error('Failed to load image:', `${getBackendURL()}/uploads/${analysis.image_path}`);
                  e.currentTarget.style.display = 'none';
                  const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                  if (fallback) fallback.style.display = 'block';
                }}
              />
              <div className="hidden text-center text-slate-500 dark:text-slate-400 mt-4">
                <p>Image could not be loaded</p>
                <p className="text-sm">The analysis was completed successfully</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Result Data */}
        {analysis.result_data && Object.keys(analysis.result_data).length > 0 && (() => {
          // Show only relevant fields without duplicates
          const displayData = [
            { label: 'Condition', value: analysis.result_data.condition_name || analysis.result_data.condition },
            { label: 'Model Name', value: analysis.model_name },
            { label: 'Analysis Method', value: analysis.result_data.method },
            { label: 'Result ID', value: analysis.id },
          ].filter(item => item.value !== null && item.value !== undefined);

          if (displayData.length === 0) return null;

          return (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
              className="card p-8 mb-6"
            >
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Analysis Details
              </h3>
              <div className="space-y-3">
                {displayData.map((item, idx) => {
                  const isResultId = item.label === 'Result ID';
                  const isModelName = item.label === 'Model Name';
                  return (
                    <div
                      key={idx}
                      className="flex justify-between items-center p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg"
                    >
                      <span className="text-slate-600 dark:text-slate-400">
                        {item.label}
                      </span>
                      <span
                        className="font-medium text-slate-900 dark:text-white"
                        style={isResultId || isModelName ? { fontFamily: 'Consolas, monospace' } : undefined}
                      >
                        {isResultId
                          ? String(item.value)
                          : isModelName
                          ? String(item.value) // Show full model name without truncation
                          : (typeof item.value === 'number'
                              ? (item.value < 1 ? `${(item.value * 100).toFixed(1)}%` : item.value.toFixed(2))
                              : typeof item.value === 'boolean'
                              ? (item.value ? 'Yes' : 'No')
                              : (typeof item.value === 'string' && item.value.length > 20
                                  ? `${item.value.substring(0, 20)}...`
                                  : String(item.value))
                            )}
                      </span>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          );
        })()}

        {/* Blockchain Verification */}
        {analysis.blockchain_hash && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className={`card p-8 mb-6 border-2 ${
              analysis.blockchain_status === 'verified'
                ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20'
                : analysis.blockchain_status === 'failed'
                ? 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20'
                : 'border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20'
            }`}
          >
            <div className="flex items-start gap-4">
              <div className={`p-3 rounded-full ${
                analysis.blockchain_status === 'verified'
                  ? 'bg-green-100 dark:bg-green-900/50'
                  : analysis.blockchain_status === 'failed'
                  ? 'bg-red-100 dark:bg-red-900/50'
                  : 'bg-amber-100 dark:bg-amber-900/50'
              }`}>
                <Shield className={`w-6 h-6 ${
                  analysis.blockchain_status === 'verified'
                    ? 'text-green-600 dark:text-green-400'
                    : analysis.blockchain_status === 'failed'
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-amber-600 dark:text-amber-400'
                }`} />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                  {analysis.blockchain_status === 'verified'
                    ? 'Blockchain Verification'
                    : analysis.blockchain_status === 'failed'
                    ? 'Verification Warning'
                    : 'Blockchain Security'}
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                  {analysis.blockchain_status === 'verified'
                    ? 'This analysis has been cryptographically secured and recorded on our blockchain for data integrity.'
                    : analysis.blockchain_status === 'failed'
                    ? 'WARNING: This analysis was previously secured, but its record cannot be found in the current blockchain. The data integrity could not be verified.'
                    : 'This analysis is marked for blockchain recording. Integrity verification is pending.'}
                </p>
                <div className="bg-white dark:bg-slate-800 rounded-lg p-3">
                  <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                    Transaction Hash
                  </p>
                  <code className="text-sm font-mono text-slate-700 dark:text-slate-300 break-all">
                    {analysis.blockchain_hash}
                  </code>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="flex flex-col sm:flex-row gap-4 mb-6"
        >
          <Link href="/analysis" className="btn btn-primary flex-1 text-center">
            Start New Analysis
          </Link>
          <Link href="/dashboard" className="btn btn-outline flex-1 text-center">
            View History
          </Link>
        </motion.div>

        {/* AI Summary Section */}
        <AISummarySection analysisId={analysisId} analysisType={analysis.analysis_type} />
        </div>
      </div>

      {/* Download Preview Modal */}
      {showDownloadModal && canvasData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-hidden">
            <div className="p-6 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                  Download Analysis Result
                </h3>
                <button
                  onClick={() => setShowDownloadModal(false)}
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                >
                  <XCircle className="w-6 h-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6 overflow-y-hidden max-h-[70vh]">
              <div className="flex gap-6">
                {/* Left side - Preview Image */}
                <div className="flex-1">
                  <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Preview
                  </h4>
                  <div className="border border-slate-200 dark:border-slate-600 rounded-lg overflow-hidden">
                    <img
                      src={canvasData.dataUrl}
                      alt="Analysis result preview"
                      className="w-full h-auto object-contain"
                    />
                  </div>
                </div>

                {/* Right side - Details */}
                <div className="w-64 space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                      Details
                    </h4>
                    <div className="space-y-3">
                      <div>
                        <h5 className="text-xs font-medium text-slate-600 dark:text-slate-400 uppercase tracking-wider mb-1">
                          Dimensions
                        </h5>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          {canvasData.width} × {canvasData.height} pixels
                        </p>
                      </div>
                      <div>
                        <h5 className="text-xs font-medium text-slate-600 dark:text-slate-400 uppercase tracking-wider mb-1">
                          File Size
                        </h5>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          ~{Math.round(canvasData.size / 1024)} KB
                        </p>
                      </div>
                      <div>
                        <h5 className="text-xs font-medium text-slate-600 dark:text-slate-400 uppercase tracking-wider mb-1">
                          Format
                        </h5>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          PNG Image
                        </p>
                      </div>
                      <div>
                        <h5 className="text-xs font-medium text-slate-600 dark:text-slate-400 uppercase tracking-wider mb-1">
                          Analysis ID
                        </h5>
                        <p className="text-sm text-slate-600 dark:text-slate-400 font-mono break-all">
                          {analysisId}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="p-6 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-700/50">
              <div className="flex justify-end">
                <button
                  onClick={handleActualDownload}
                  className="btn btn-primary inline-flex items-center gap-2 px-8"
                >
                  <Download className="w-4 h-4" />
                  Download PNG
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// AI Summary Component
function AISummarySection({ analysisId, analysisType }: { analysisId: string; analysisType: string }) {
  const [summary, setSummary] = useState<string | null>(null);
  const [isLoadingSummary, setIsLoadingSummary] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Auto-load summary on component mount
  useEffect(() => {
    const loadSummary = async () => {
      try {
        setIsLoadingSummary(true);
        setError(null);
        const response = await chatApi.getAnalysisSummary(analysisId);
        setSummary(response.summary);
      } catch (err) {
        console.error('Failed to load summary:', err);
        setError('Unable to generate summary at this time. Please try again later or chat with Dr. Hygieia for more information.');
      } finally {
        setIsLoadingSummary(false);
      }
    };

    loadSummary();
  }, [analysisId]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="card overflow-hidden"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-primary-500 to-emerald-500 p-6 text-white">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-white/20 flex items-center justify-center">
            <img src="/hygieia_color.png" alt="Dr. Hygieia" className="w-9 h-9" />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              Dr. Hygieia AI Assistant
            </h3>
            <p className="text-white/80 text-sm">
              Get AI-powered insights about your {getAnalysisTypeLabel(analysisType).toLowerCase()} results
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {isLoadingSummary ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            <span className="ml-3 text-slate-600 dark:text-slate-400">
              Dr. Hygieia is analyzing your results...
            </span>
          </div>
        ) : error ? (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 mb-4">
            <p className="text-red-700 dark:text-red-300">
              {error}
            </p>
          </div>
        ) : summary ? (
          <div>
            <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-6 mb-4">
              <div className="max-w-none text-slate-700 dark:text-slate-300">
                <ReactMarkdown
                  components={{
                    h1: ({ node, ...props }: any) => <h1 className="text-2xl font-bold mb-4 text-slate-900 dark:text-white" {...props} />,
                    h2: ({ node, ...props }: any) => <h2 className="text-xl font-bold mb-3 text-slate-900 dark:text-white mt-4" {...props} />,
                    h3: ({ node, ...props }: any) => <h3 className="text-lg font-semibold mb-2 text-slate-900 dark:text-white mt-3" {...props} />,
                    p: ({ node, ...props }: any) => <p className="mb-3 text-slate-700 dark:text-slate-300 leading-relaxed" {...props} />,
                    ul: ({ node, ...props }: any) => <ul className="list-disc list-inside mb-3 text-slate-700 dark:text-slate-300 space-y-1" {...props} />,
                    ol: ({ node, ...props }: any) => <ol className="list-decimal list-inside mb-3 text-slate-700 dark:text-slate-300 space-y-1" {...props} />,
                    li: ({ node, ...props }: any) => <li className="mb-1" {...props} />,
                    strong: ({ node, ...props }: any) => <strong className="font-semibold text-slate-900 dark:text-white" {...props} />,
                    em: ({ node, ...props }: any) => <em className="italic text-slate-700 dark:text-slate-300" {...props} />,
                    code: ({ node, inline, ...props }: any) => inline ? 
                      <code className="bg-slate-200 dark:bg-slate-700 px-2 py-1 rounded text-sm font-mono text-slate-900 dark:text-white" {...props} /> :
                      <code className="bg-slate-200 dark:bg-slate-700 px-3 py-2 rounded-lg block mb-3 text-sm font-mono text-slate-900 dark:text-white overflow-x-auto" {...props} />,
                    blockquote: ({ node, ...props }: any) => <blockquote className="border-l-4 border-primary-500 pl-4 italic text-slate-600 dark:text-slate-400 mb-3" {...props} />,
                    a: ({ node, ...props }: any) => <a className="text-primary-600 dark:text-primary-400 hover:underline" {...props} />,
                  } as any}
                >
                  {summary}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        ) : null}

        {/* Chat CTA */}
        <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-center sm:text-left">
              <h4 className="font-semibold text-slate-900 dark:text-white">
                Have questions about your results?
              </h4>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Chat with Dr. Hygieia for personalized explanations
              </p>
            </div>
            <button
              onClick={async () => {
                try {
                  // Check for an existing chat session for this analysis
                  const sessionsResp = await chatApi.getSessions({ per_page: 50 });
                  const existing = sessionsResp.sessions?.find((s: any) => s.analysis_id === analysisId);
                  if (existing) {
                    // Redirect to existing session
                    window.location.href = `/chat?session_id=${existing.id}`;
                  } else {
                    // Create a new session server-side and redirect to it immediately
                    const created = await chatApi.createSession({ analysis_id: analysisId });
                    const newId = created.session?.id;
                    if (newId) {
                      window.location.href = `/chat?session_id=${newId}`;
                    } else {
                      // Fallback to opening chat page which can also create session
                      window.location.href = `/chat?analysis_id=${analysisId}`;
                    }
                  }
                } catch (e) {
                  // Fallback
                  window.location.href = `/chat?analysis_id=${analysisId}`;
                }
              }}
              className="btn btn-outline btn-md inline-flex items-center gap-2 whitespace-nowrap"
            >
              <MessageCircle className="w-4 h-4" />
              Chat with Dr. Hygieia
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
