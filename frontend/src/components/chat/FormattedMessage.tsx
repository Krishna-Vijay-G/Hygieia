'use client';

import React from 'react';
import { cn } from '@/lib/utils';

interface FormattedMessageProps {
  content: string;
}

export function FormattedMessage({ content }: FormattedMessageProps) {
  // Parse the content and render with proper formatting
  const parseContent = (text: string) => {
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let orderedListItems: string[] = [];
    let unorderedListItems: string[] = [];

    lines.forEach((line, idx) => {
      const trimmed = line.trim();

      // Handle ### headers
      if (trimmed.startsWith('### ')) {
        // Flush any accumulated lists
        flushLists();
        elements.push(
          <h3 key={`header-${idx}`} className="text-lg font-bold mt-4 mb-2">
            {renderInlineFormatting(trimmed.replace('### ', ''))}
          </h3>
        );
      }
      // Handle numbered lists (1. 2. 3.)
      else if (/^\d+\./.test(trimmed)) {
        // Flush unordered list if exists
        if (unorderedListItems.length > 0) {
          flushUnorderedList();
        }
        orderedListItems.push(trimmed);
      }
      // Handle bullet points (- item or * item)
      else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
        // Flush ordered list if exists
        if (orderedListItems.length > 0) {
          flushOrderedList();
        }
        unorderedListItems.push(trimmed);
      }
      else {
        // Handle empty lines
        if (!trimmed) {
          // If currently building a list, treat blank line as part of the list
          // (do not flush) so numbering remains continuous even when the AI
          // inserts blank lines between items. Otherwise add spacing.
          if (orderedListItems.length > 0 || unorderedListItems.length > 0) {
            // keep accumulating list items
            return;
          }
          elements.push(<div key={`space-${idx}`} className="h-2" />);
        }
        // Handle bold headers (lines ending with :)
        else if (trimmed.endsWith(':') && trimmed.length < 100) {
          elements.push(
            <p key={idx} className="font-semibold text-sm mt-3 mb-2">
              {renderInlineFormatting(trimmed)}
            </p>
          );
        }
        // Handle regular paragraphs
        else if (trimmed) {
          // Flush any accumulated list items before starting a paragraph
          flushLists();
          elements.push(
            <p key={idx} className="text-sm leading-relaxed mb-2">
              {renderInlineFormatting(trimmed)}
            </p>
          );
        }
      }
    });

    // Flush any remaining lists
    flushLists();

    function flushLists() {
      flushOrderedList();
      flushUnorderedList();
    }

    function flushOrderedList() {
      if (orderedListItems.length > 0) {
        // Render explicit numbering to avoid CSS/list-style issues
        elements.push(
          <div key={`ordered-list-${elements.length}`} className="mb-3 ml-2 space-y-1">
            {orderedListItems.map((item, i) => (
              <div key={i} className="flex items-start gap-3">
                <span className="text-sm font-semibold text-slate-500 dark:text-slate-400 mt-0.5">{i + 1}.</span>
                <div className="text-sm leading-relaxed">
                  {renderInlineFormatting(item.replace(/^\d+\.\s*/, ''))}
                </div>
              </div>
            ))}
          </div>
        );
        orderedListItems = [];
      }
    }

    function flushUnorderedList() {
      if (unorderedListItems.length > 0) {
        elements.push(
          <ul key={`unordered-list-${elements.length}`} className="list-disc list-inside mb-3 space-y-1 ml-2">
            {unorderedListItems.map((item, i) => (
              <li key={i} className="text-sm leading-relaxed">
                {renderInlineFormatting(item.replace(/^[-*]\s*/, ''))}
              </li>
            ))}
          </ul>
        );
        unorderedListItems = [];
      }
    }

    return elements;
  };

  const renderInlineFormatting = (text: string) => {
    // Handle **bold** and *italic* text
    // Split on both patterns but keep them in order
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let key = 0;

    while (remaining.length > 0) {
      // Try to match **bold**
      const boldMatch = remaining.match(/\*\*([^*]+)\*\*/);
      // Try to match *italic* (but not **)
      const italicMatch = remaining.match(/(?<!\*)\*([^*]+)\*(?!\*)/);

      // Determine which match comes first
      const boldIndex = boldMatch ? remaining.indexOf(boldMatch[0]) : Infinity;
      const italicIndex = italicMatch ? remaining.indexOf(italicMatch[0]) : Infinity;

      if (boldIndex < italicIndex) {
        // Process bold first
        if (boldIndex > 0) {
          parts.push(remaining.substring(0, boldIndex));
        }
        parts.push(
          <strong key={key++} className="font-semibold">
            {boldMatch![1]}
          </strong>
        );
        remaining = remaining.substring(boldIndex + boldMatch![0].length);
      } else if (italicIndex < Infinity) {
        // Process italic first
        if (italicIndex > 0) {
          parts.push(remaining.substring(0, italicIndex));
        }
        parts.push(
          <em key={key++} className="italic">
            {italicMatch![1]}
          </em>
        );
        remaining = remaining.substring(italicIndex + italicMatch![0].length);
      } else {
        // No more formatting
        parts.push(remaining);
        break;
      }
    }

    return parts;
  };

  return (
    <div className="space-y-2">
      {parseContent(content)}
    </div>
  );
}
