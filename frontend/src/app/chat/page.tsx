'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import Link from 'next/link';
import { 
  Send, 
  Plus, 
  MessageSquare, 
  Trash2, 
  Edit2,
  Check,
  X,
  Loader2,
  Bot,
  ChevronLeft,
  Menu,
  Sparkles,
  ChevronDown,
  Calendar,
  Target,
  TrendingUp,
  ExternalLink
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { chatApi } from '@/lib/api';
import { cn, getImageURL, getBackendURL, formatDateTime } from '@/lib/utils';
import { FormattedMessage } from '@/components/chat/FormattedMessage';

interface Message {
  id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

interface Session {
  id: string;
  title: string;
  analysis_id?: string;
  context_type: string;
  message_count: number;
  created_at: string;
  updated_at: string;
  messages?: Message[];
  user?: {
    id: string;
    username: string;
    first_name: string;
    last_name: string;
    full_name: string;
  };
}

export default function ChatPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();
  
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState('');
  const [editingTitle, setEditingTitle] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [optimisticMessages, setOptimisticMessages] = useState<Message[]>([]);
  const [showContextDropdown, setShowContextDropdown] = useState(false);
  const [usernameFilter, setUsernameFilter] = useState<string>('all');
  const [usernameSearch, setUsernameSearch] = useState<string>('');
  const [showUsernameDropdown, setShowUsernameDropdown] = useState(false);
  const usernameDropdownRef = useRef<HTMLDivElement>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const contextDropdownRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change (only scrolls the chat container, not the page)
  const scrollToBottom = useCallback(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, []);

  // Close context dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (contextDropdownRef.current && !contextDropdownRef.current.contains(event.target as Node)) {
        setShowContextDropdown(false);
      }
    };

    if (showContextDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showContextDropdown]);

  // Check auth
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  // Fetch sessions
  const { data: sessionsData, isLoading: sessionsLoading } = useQuery({
    queryKey: ['chat-sessions'],
    queryFn: () => chatApi.getSessions({ per_page: 50 }),
    enabled: isAuthenticated,
  });

  // Fetch active session
  const { data: sessionData, isLoading: sessionLoading } = useQuery({
    queryKey: ['chat-session', activeSessionId],
    queryFn: () => chatApi.getSession(activeSessionId!),
    enabled: !!activeSessionId,
    staleTime: 0,
  });

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (data?: { analysis_id?: string }) => chatApi.createSession(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      setActiveSessionId(data.session.id);
    },
  });

  // Handle analysis_id from URL params
  useEffect(() => {
    const analysisId = searchParams.get('analysis_id');
    if (analysisId && isAuthenticated && sessionsData && !activeSessionId) {
      // Check if a session for this analysis already exists
      const existingSession = sessionsData.sessions?.find(
        (s: Session) => s.analysis_id === analysisId
      );

      if (existingSession) {
        // Use existing session
        setActiveSessionId(existingSession.id);
      } else {
        // Create a new session with analysis context
        createSessionMutation.mutate({ analysis_id: analysisId });
      }
    }
  }, [searchParams, isAuthenticated, sessionsData, activeSessionId, createSessionMutation]);

  // If a session_id is provided in the URL, open it directly
  useEffect(() => {
    const sessionIdParam = searchParams.get('session_id');
    if (sessionIdParam && sessionIdParam !== activeSessionId) {
      setActiveSessionId(sessionIdParam);
    }
  }, [searchParams]);

  // Keep the URL in sync when user clicks a session: update ?session_id= without reloading
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (activeSessionId) {
      const currentParam = searchParams.get('session_id');
      if (currentParam !== activeSessionId) {
        // replace to avoid pushing history on every click
        router.replace(`/chat?session_id=${activeSessionId}`, { scroll: false });
      }
    } else {
      const hasParams = searchParams.get('session_id') || searchParams.get('analysis_id');
      if (hasParams) {
        router.replace('/chat', { scroll: false });
      }
    }
  }, [activeSessionId, router, searchParams]);

  // Check session access permissions
  useEffect(() => {
    if (!authLoading && isAuthenticated && sessionData && activeSessionId) {
      const isOwner = sessionData.is_owner;
      const isAdmin = user?.is_admin;
      
      if (!isOwner && !isAdmin) {
        // Regular user trying to access someone else's session
        router.push('/access-denied');
      }
    }
  }, [authLoading, isAuthenticated, sessionData, activeSessionId, user, router]);

  // Auto-scroll when messages or typing state changes
  useEffect(() => {
    scrollToBottom();
  }, [sessionData, optimisticMessages, isTyping, scrollToBottom]);

  // Clear optimistic messages when session changes
  useEffect(() => {
    setOptimisticMessages([]);
  }, [activeSessionId]);

  // Update session mutation
  const updateSessionMutation = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => 
      chatApi.updateSession(id, { title }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      setEditingTitle(null);
    },
  });

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (id: string) => chatApi.deleteSession(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      if (activeSessionId === deleteSessionMutation.variables) {
        setActiveSessionId(null);
      }
    },
  });

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: ({ sessionId, content }: { sessionId: string; content: string }) =>
      chatApi.sendMessage(sessionId, content),
    onMutate: ({ content }) => {
      // Create optimistic user message
      const optimisticMessage: Message = {
        id: `temp-${Date.now()}`,
        session_id: activeSessionId!,
        role: 'user',
        content,
        created_at: new Date().toISOString(),
      };
      setOptimisticMessages([optimisticMessage]);
      setIsTyping(true);
      setInputMessage('');
    },
    onSuccess: async () => {
      // Wait for the query to refetch and show the real messages
      await queryClient.invalidateQueries({ queryKey: ['chat-session', activeSessionId] });
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      
      // Give the query a moment to refetch
      await new Promise(resolve => setTimeout(resolve, 300));
      setIsTyping(false);
      setOptimisticMessages([]);
    },
    onError: () => {
      setIsTyping(false);
      setOptimisticMessages([]);
    },
  });

  const handleSendMessage = useCallback(() => {
    if (!inputMessage.trim() || !activeSessionId || sendMessageMutation.isPending) return;
    
    sendMessageMutation.mutate({
      sessionId: activeSessionId,
      content: inputMessage.trim(),
    });
  }, [inputMessage, activeSessionId, sendMessageMutation]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleNewChat = () => {
    createSessionMutation.mutate({});
  };

  const handleEditTitle = (session: Session) => {
    setEditingTitle(session.id);
    setNewTitle(session.title);
  };

  const handleSaveTitle = (sessionId: string) => {
    if (newTitle.trim()) {
      updateSessionMutation.mutate({ id: sessionId, title: newTitle.trim() });
    } else {
      setEditingTitle(null);
    }
  };

  const sessions = sessionsData?.sessions || [];
  
  // Extract unique usernames for filter (only for admins).
  // Use a reduce-based approach instead of Set iteration to avoid
  // TypeScript downlevel iteration issues on older targets.
  const uniqueUsernames: string[] = user?.is_admin
    ? sessions.reduce((acc: string[], s: Session) => {
        const uname = s.user?.username;
        if (uname && !acc.includes(uname)) acc.push(uname);
        return acc;
      }, [] as string[])
      .sort()
      .filter((u: string) => u.toLowerCase().includes(usernameSearch.toLowerCase()))
    : [];
  
  // Filter sessions based on selected username
  const filteredSessions = user?.is_admin && usernameFilter !== 'all'
    ? sessions.filter((session: Session) => session.user?.username === usernameFilter)
    : sessions;

  // If the current usernameFilter no longer exists in the filtered unique list, reset to 'all'
  useEffect(() => {
    if (user?.is_admin && usernameFilter !== 'all' && !uniqueUsernames.includes(usernameFilter)) {
      setUsernameFilter('all');
    }
  }, [usernameSearch, uniqueUsernames, usernameFilter, user?.is_admin]);

  // Combine real messages with optimistic messages
  const allMessages = [
    ...(sessionData?.session?.messages || []),
    ...optimisticMessages,
  ];
  useEffect(() => {
    if (user?.is_admin && usernameFilter !== 'all' && activeSessionId) {
      const isActiveSessionInFilter = filteredSessions.some((session: Session) => session.id === activeSessionId);
      if (!isActiveSessionInFilter) {
        setActiveSessionId(null);
      }
    }
  }, [usernameFilter, filteredSessions, activeSessionId, user?.is_admin]);

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-4rem)] bg-slate-50 dark:bg-slate-900">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            className="w-72 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 flex flex-col"
          >
            {/* Sidebar Header */}
            <div className="p-4 border-b border-slate-200 dark:border-slate-700">
              <button
                onClick={handleNewChat}
                disabled={createSessionMutation.isPending}
                className="w-full btn-primary btn-md flex items-center justify-center gap-2"
              >
                {createSessionMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Plus className="w-4 h-4" />
                )}
                New Chat
              </button>
            </div>

            {/* Username Filter (Admin Only) */}
            {user?.is_admin && uniqueUsernames.length > 0 && (
              <div className="p-4 border-b border-slate-200 dark:border-slate-700">
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-2">
                  Filter by User
                </label>
                <div className="relative" ref={usernameDropdownRef}>
                  <button
                    onClick={() => setShowUsernameDropdown(!showUsernameDropdown)}
                    className="w-full text-left px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg flex items-center justify-between gap-2"
                  >
                      <span className="text-sm">
                      {usernameFilter === 'all' ? `All Users (${sessions.length})` : `${usernameFilter} (${sessions.filter((s: Session) => s.user?.username === usernameFilter).length})`}
                    </span>
                    <ChevronDown className={`w-4 h-4 transition-transform ${showUsernameDropdown ? 'rotate-180' : ''}`} />
                  </button>

                  <AnimatePresence>
                    {showUsernameDropdown && (
                      <motion.div
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.12 }}
                        className="absolute z-20 left-0 right-0 mt-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg overflow-hidden"
                      >
                        <div className="p-2">
                          <input
                            type="text"
                            placeholder="Search users..."
                            value={usernameSearch}
                            onChange={(e) => setUsernameSearch(e.target.value)}
                            className="w-full mb-2 px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 focus:outline-none"
                            autoFocus
                          />
                          <div className="max-h-40 overflow-y-auto">
                            <button
                              onClick={() => { setUsernameFilter('all'); setShowUsernameDropdown(false); setUsernameSearch(''); }}
                              className="w-full text-left px-2 py-1 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
                            >
                              All Users ({sessions.length})
                            </button>
                            {uniqueUsernames.map((username: string) => {
                              const userChats = sessions.filter((s: Session) => s.user?.username === username);
                              return (
                                <button
                                  key={username}
                                  onClick={() => { setUsernameFilter(username); setShowUsernameDropdown(false); }}
                                  className="w-full text-left px-2 py-1 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
                                >
                                  {username} ({userChats.length})
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            )}

            {/* Sessions List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {sessionsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
                </div>
              ) : filteredSessions.length === 0 ? (
                <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                  <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">
                    {usernameFilter === 'all' ? 'No conversations yet' : `No conversations found for ${usernameFilter}`}
                  </p>
                  {usernameFilter !== 'all' && (
                    <button
                      onClick={() => setUsernameFilter('all')}
                      className="text-xs text-primary-600 dark:text-primary-400 hover:underline mt-1"
                    >
                      Show all conversations
                    </button>
                  )}
                </div>
              ) : (
                filteredSessions.map((session: Session) => (
                  <div
                    key={session.id}
                    className={cn(
                      'group relative rounded-lg p-3 cursor-pointer transition-colors',
                      activeSessionId === session.id
                        ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                        : 'hover:bg-slate-100 dark:hover:bg-slate-700/50 text-slate-700 dark:text-slate-300'
                    )}
                    onClick={() => setActiveSessionId(session.id)}
                  >
                    {editingTitle === session.id ? (
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={newTitle}
                          onChange={(e) => setNewTitle(e.target.value)}
                          className="flex-1 min-w-0 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-sm"
                          style={{ maxWidth: 'calc(100% - 72px)' }}
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveTitle(session.id);
                            if (e.key === 'Escape') setEditingTitle(null);
                          }}
                          onClick={(e) => e.stopPropagation()}
                        />
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSaveTitle(session.id);
                          }}
                          className="p-1 hover:bg-green-100 dark:hover:bg-green-900/30 rounded"
                        >
                          <Check className="w-4 h-4 text-green-600" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingTitle(null);
                          }}
                          className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded"
                        >
                          <X className="w-4 h-4 text-red-600" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-start gap-2">
                          <MessageSquare className="w-4 h-4 mt-0.5 flex-shrink-0" />
                          <div className="flex-1 min-w-0 pr-10">
                            <p className="text-sm font-medium truncate">{session.title}</p>
                            <div className="flex items-center gap-2 mt-0.5">
                              <p className="text-xs text-slate-500 dark:text-slate-400">
                                {session.message_count} messages
                              </p>
                              {user?.is_admin && session.user && session.user.id !== user.id && (
                                <span className="text-xs bg-slate-100 dark:bg-slate-600 px-2 py-0.5 rounded-full text-slate-600 dark:text-slate-300">
                                  {session.user.username}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 flex items-center gap-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditTitle(session);
                            }}
                            className="p-1 hover:bg-slate-200 dark:hover:bg-slate-600 rounded"
                          >
                            <Edit2 className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteSessionMutation.mutate(session.id);
                            }}
                            className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded text-red-600"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                ))
              )}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <div className="h-14 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 flex items-center px-4 gap-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
          >
            {sidebarOpen ? <ChevronLeft className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
              <img src="/hygieia_color.png" alt="Dr. Hygieia" className="w-8 h-8" />
            </div>
            <div>
              <h1 className="font-semibold text-slate-900 dark:text-white">Dr. Hygieia</h1>
              <div className="flex items-center gap-3">
                <p className="text-xs text-slate-500 dark:text-slate-400">AI Health Assistant</p>
                {sessionData?.session?.title && ( 
                  <p className="text-xs text-slate-500 dark:text-slate-400 truncate max-w-xs">
                    - {sessionData.session.title} -
                  </p>
                )}
              </div>
            </div>
          </div>

          {sessionData?.session?.analysis && (
            <div className="ml-auto relative" ref={contextDropdownRef}>
              <button
                onClick={() => setShowContextDropdown(!showContextDropdown)}
                className="flex items-center gap-2 px-3 py-1 rounded-lg text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
              >
                <Sparkles className="w-4 h-4 text-primary-500" />
                <span className="capitalize">{sessionData.session.analysis.analysis_type.replace('-', ' ')}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${showContextDropdown ? 'rotate-180' : ''}`} />
              </button>

              {/* Context Dropdown */}
              <AnimatePresence>
                {showContextDropdown && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-full mt-2 w-96 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl shadow-lg z-50"
                  >
                    <div className="p-4 border-b border-slate-200 dark:border-slate-700">
                      <h3 className="font-semibold text-slate-900 dark:text-white capitalize mb-1">
                        {sessionData.session.analysis.analysis_type.replace('-', ' ')}
                      </h3>
                      {sessionData.session.analysis.created_at && (
                        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                          <Calendar className="w-3 h-3" />
                          {formatDateTime(sessionData.session.analysis.created_at)}
                        </div>
                      )}
                    </div>

                    {/* Results Summary */}
                    <div className="p-4 border-b border-slate-200 dark:border-slate-700">
                      <div className="space-y-3">
                        {sessionData.session.analysis.risk_level && (
                          <div>
                            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 mb-1">
                              <Target className="w-3 h-3" />
                              <span>Risk Level</span>
                            </div>
                            <p className="text-sm font-medium text-slate-900 dark:text-white capitalize">
                              {sessionData.session.analysis.risk_level}
                            </p>
                          </div>
                        )}
                        {sessionData.session.analysis.confidence !== null && sessionData.session.analysis.confidence !== undefined && (
                          <div>
                            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 mb-1">
                              <TrendingUp className="w-3 h-3" />
                              <span>Confidence</span>
                            </div>
                            <p className="text-sm font-medium text-slate-900 dark:text-white">
                              {(sessionData.session.analysis.confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Input Parameters / Analyzed Image */}
                    {((sessionData.session.analysis.input_data && Object.keys(sessionData.session.analysis.input_data).length > 0) || 
                      (sessionData.session.analysis.analysis_type === 'skin-diagnosis' && sessionData.session.analysis.image_path)) && (() => {
                      const inputData = sessionData.session.analysis.input_data || {};
                      // Prefer 'actual_' fields when present and hide numeric/code counterparts.
                      const visibleEntries = Object.entries(inputData).filter(([k]) => {
                        if (k.startsWith('actual_')) return true;
                        const counterpart = `actual_${k}`;
                        if (inputData[counterpart] !== undefined) return false;
                        return true;
                      });

                      // Sort so actual_ entries appear first
                      visibleEntries.sort(([a], [b]) => (a.startsWith('actual_') === b.startsWith('actual_') ? 0 : a.startsWith('actual_') ? -1 : 1));

                      return (
                        <div className="p-4 border-b border-slate-200 dark:border-slate-700 max-h-64 overflow-y-auto">
                          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase mb-2">
                            {sessionData.session.analysis.analysis_type === 'skin-diagnosis' ? 'Analyzed Image' : 'Input Parameters'}
                          </p>
                          <div className="space-y-2">
                            {/* For skin lesion analyses, show the input image and identified class */}
                            {sessionData.session.analysis.analysis_type === 'skin-diagnosis' && (() => {
                              const analysis = sessionData.session.analysis as any;
                              const imagePath = analysis.image_path;
                              if (!imagePath) return null;

                              const src = `${getBackendURL()}/uploads/${imagePath}`;
                              const result = analysis.result || analysis.result_data || {};
                              const identifiedClass = result.condition || result.condition_name || result.prediction || 'Unknown';
                              const confidence = result.confidence;

                              return (
                                <div className="mb-3">
                                  <div className="flex items-start gap-4">
                                    <a href={src} target="_blank" rel="noreferrer" className="block flex-shrink-0">
                                      <div className="relative aspect-video w-48 rounded-md overflow-hidden border-2 border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800">
                                        <img
                                          src={src}
                                          alt="Analyzed lesion"
                                          className="w-full h-full object-cover"
                                          onError={(e: any) => { 
                                            console.error('Failed to load image:', src);
                                            e.currentTarget.style.display = 'none'; 
                                          }}
                                        />
                                      </div>
                                    </a>
                                    <div className="flex-1 min-w-0 py-1">
                                      <div className="text-sm font-semibold text-slate-900 dark:text-white mb-1">{identifiedClass}</div>
                                      <a 
                                        href={src} 
                                        target="_blank" 
                                        rel="noreferrer"
                                        className="text-xs text-primary-600 dark:text-primary-400 hover:underline mt-1 inline-block"
                                      >
                                        View full size →
                                      </a>
                                    </div>
                                  </div>
                                </div>
                              );
                            })()}
                            {visibleEntries.map(([key, value]) => {
                              // Normalize key (remove actual_ and spaces, use underscore-separated lower-case)
                              const rawKey = key.replace(/^actual_/i, '');
                              const normalizedKey = rawKey.replace(/\s+/g, '_').toLowerCase();

                              // Friendly field labels for known fields
                              const FIELD_LABELS: Record<string, string> = {
                                age: 'Age',
                                bmi: 'BMI',
                                density: 'Breast Density',
                                race: 'Race/Ethnicity',
                                age_menarche: 'Age at Menarche',
                                age_first_birth: 'Age at First Birth',
                                agefirst: 'Age at First Birth',
                                brstproc: 'Previous Biopsy',
                                hrt: 'HRT Use',
                                family_hx: 'Family History',
                                menopaus: 'Menopausal Status',
                                nrelbc: 'Number of Relatives with Breast Cancer',
                                // add more domain-specific keys here as needed
                              };

                              const displayKey = FIELD_LABELS[normalizedKey] || rawKey.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

                              // Format value for readability (map 0/1 to Yes/No, gender numeric to Male/Female, units)
                              const analysisType = sessionData.session.analysis.analysis_type || '';
                              let displayValue: string;

                              if (typeof value === 'boolean') {
                                displayValue = value ? 'Yes' : 'No';
                              } else if (value === 0 || value === 1 || value === '0' || value === '1') {
                                const keyLower = normalizedKey;
                                if (keyLower.includes('gender') || keyLower === 'sex') {
                                  displayValue = (value === 1 || value === '1') ? 'Male' : 'Female';
                                } else if (analysisType === 'heart-prediction' && (keyLower === 'gender' || keyLower === 'sex')) {
                                  displayValue = (value === 1 || value === '1') ? 'Male' : 'Female';
                                } else {
                                  // Generic binary -> Yes/No
                                  displayValue = (value === 1 || value === '1') ? 'Yes' : 'No';
                                }
                              } else if (normalizedKey === 'nrelbc' && (typeof value === 'number' || /^\d+$/.test(String(value)))) {
                                const num = Number(value);
                                displayValue = `${num} relative${num === 1 ? '' : 's'} with breast cancer`;
                              } else if ((normalizedKey === 'age' || normalizedKey === 'bmi') && !isNaN(Number(value))) {
                                // numeric display with minimal decimals
                                const num = Number(value);
                                displayValue = Number.isInteger(num) ? String(num) : num.toFixed(1);
                              } else {
                                displayValue = String(value ?? 'N/A');
                              }

                              return (
                                <div key={key} className="text-xs">
                                  <span className="text-slate-600 dark:text-slate-400">{displayKey}:</span>
                                  <span className="ml-2 font-medium text-slate-900 dark:text-white">{displayValue}</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })()}

                    {/* Action Buttons */}
                    <div className="p-4 flex gap-2">
                      <Link
                        href={`/analysis/result/${sessionData.session.analysis.id}`}
                        className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-primary-500 hover:bg-primary-600 text-white text-sm font-medium rounded-lg transition-colors"
                      >
                        <ExternalLink className="w-4 h-4" />
                        View Details
                      </Link>
                      <button
                        onClick={() => setShowContextDropdown(false)}
                        className="flex-1 px-3 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 text-sm font-medium rounded-lg transition-colors"
                      >
                        Close
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>

        {/* Messages Area */}
        <div 
          ref={messagesContainerRef}
          className="flex-1 overflow-y-auto p-4 space-y-4"
        >
          {!activeSessionId ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <div className="w-20 h-20 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center mb-4">
                <img src="/hygieia_color.png" alt="Dr. Hygieia" className="w-20 h-20" />
              </div>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
                Welcome to Dr. Hygieia
              </h2>
              <p className="text-slate-600 dark:text-slate-400 max-w-md mb-6">
                Your AI health assistant. Ask me about your analysis results, health questions, 
                or anything related to your wellbeing.
              </p>
              <button
                onClick={handleNewChat}
                disabled={createSessionMutation.isPending}
                className="btn-primary btn-lg"
              >
                {createSessionMutation.isPending ? (
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                ) : (
                  <MessageSquare className="w-5 h-5 mr-2" />
                )}
                Start a Conversation
              </button>
            </div>
          ) : sessionLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            </div>
          ) : (
            <>
              {allMessages.map((message: Message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    'flex gap-3',
                    message.role === 'user' ? 'flex-row-reverse' : ''
                  )}
                >
                  {/* Avatar */}
                  <div className={cn(
                    'w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center',
                    message.role === 'user'
                      ? 'bg-primary-500'
                      : 'bg-primary-100 dark:bg-primary-900/30'
                  )}>
                    {message.role === 'user' ? (
                      user?.avatar_url ? (
                        <img
                          src={getImageURL(user.avatar_url)}
                          alt={user.first_name}
                          className="w-8 h-8 rounded-full object-cover"
                        />
                      ) : (
                        <span className="text-white text-sm font-medium">
                          {user?.first_name?.[0]?.toUpperCase()}
                        </span>
                      )
                    ) : (
                      <img src="/hygieia_color.png" alt="Dr. Hygieia" className="w-8 h-8" />
                    )}
                  </div>

                  {/* Message Content + timestamp (timestamp rendered below bubble) */}
                  <div className={cn('flex flex-col max-w-[70%]', message.role === 'user' ? 'items-end' : 'items-start')}>
                    <div className={cn(
                      'rounded-2xl px-4 py-3',
                      message.role === 'user'
                        ? 'bg-primary-500 text-white rounded-tr-sm'
                        : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-tl-sm'
                    )}>
                      {message.role === 'user' ? (
                        <p className="text-sm whitespace-pre-wrap text-white">
                          {message.content}
                        </p>
                      ) : (
                        <div className="dark:text-slate-300 text-slate-700">
                          <FormattedMessage 
                            content={message.content}
                          />
                        </div>
                      )}
                    </div>
                    <div className={cn(
                      'text-xs mt-2',
                      'text-slate-500 dark:text-slate-400'
                    )}>
                      {formatDateTime(message.created_at)}
                    </div>
                  </div>
                </motion.div>
              ))}

              {/* Typing Indicator */}
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3"
                >
                  <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                    <img src="/hygieia_color.png" alt="Dr. Hygieia" className="w-5 h-5" />
                  </div>
                  <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        {activeSessionId && (
          <div className="p-4 bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
            {(() => {
              const isOwner = sessionData?.is_owner;
              const isAdmin = user?.is_admin;
              const sessionOwner = sessionData?.session_owner;
              const analysis = sessionData?.session?.analysis;
              
              // If admin viewing someone else's session, show read-only message
              if (isAdmin && !isOwner) {
                const ownerName = sessionOwner ? `${sessionOwner.first_name} ${sessionOwner.last_name}` : 'Unknown User';
                const ownerUsername = sessionOwner?.username || 'unknown';
                
                let message = '';
                let linkHref = '';
                let linkText = '';
                
                if (analysis) {
                  // Analysis-based chat
                  const analysisType = analysis.analysis_type.replace('-', ' ');
                  const createdAt = analysis.created_at
                    ? new Date(analysis.created_at).toLocaleString(undefined, { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true })
                    : 'Unknown date';

                  message = `This chat belongs to ${ownerName} (${ownerUsername}) of their ${analysisType} analysis on ${createdAt}.`;
                  linkHref = `/analysis/result/${analysis.id}`;
                  linkText = 'View Analysis Results';
                } else {
                  // General chat
                  const createdAt = sessionData?.session?.created_at
                    ? new Date(sessionData.session.created_at).toLocaleString(undefined, { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true })
                    : 'Unknown date';
                  message = `This chat belongs to ${ownerName} (${ownerUsername}) on ${createdAt} - General Chat.`;
                  linkHref = '';
                  linkText = '';
                }
                
                return (
                  <div className="text-center py-2">
                    <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-2 border border-slate-200 dark:border-slate-600">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-slate-600 dark:text-slate-400 mb-0 flex-1 truncate">{message}</p>
                        {linkHref && (
                          <Link
                            href={linkHref}
                            className="ml-3 inline-flex items-center gap-2 px-2 py-1 text-xs bg-primary-500 hover:bg-primary-600 text-white font-medium rounded transition-colors"
                          >
                            <ExternalLink className="w-4 h-4" />
                            {linkText}
                          </Link>
                        )}
                      </div>
                    </div>
                  </div>
                );
              }
              
              // Normal input for owners
              return (
                <div className="flex items-end gap-3">
                  <div className="flex-1 relative">
                    <textarea
                      ref={inputRef}
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Type your message..."
                      rows={1}
                      className="w-full resize-none rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 px-4 py-3 pr-12 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 focus:outline-none transition-all duration-200"
                      style={{ maxHeight: '150px' }}
                    />
                  </div>
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || sendMessageMutation.isPending}
                    className={cn(
                      'p-3 rounded-xl transition-all duration-200',
                      inputMessage.trim() && !sendMessageMutation.isPending
                        ? 'bg-primary-500 text-white hover:bg-primary-600 shadow-lg shadow-primary-500/25'
                        : 'bg-slate-200 dark:bg-slate-700 text-slate-400 cursor-not-allowed'
                    )}
                  >
                    {sendMessageMutation.isPending ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Send className="w-5 h-5" />
                    )}
                  </button>
                </div>
              );
            })()}
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-2 text-center">
              Dr. Hygieia provides educational information only. Always consult healthcare professionals for medical advice.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
