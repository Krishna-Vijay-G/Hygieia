'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Users, 
  Shield, 
  ShieldOff,
  Search,
  UserCheck,
  UserX,
  Trash2,
  Activity,
  Mail,
  Phone,
  Calendar,
  AlertTriangle,
  Heart,
  Droplets,
  Scan,
  Ribbon,
  ChevronUp,
  ChevronDown,
  Filter
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { usersApi, analysisApi } from '@/lib/api';
import { formatDateTime, getImageURL } from '@/lib/utils';
import Link from 'next/link';

interface UserData {
  id: string;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  phone: string | null;
  avatar_url: string | null;
  is_admin: boolean;
  is_owner: boolean;
  is_active: boolean;
  created_at: string;
  analysis_count: number;
}

export default function UsersPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [userToDelete, setUserToDelete] = useState<UserData | null>(null);
  const [sortField, setSortField] = useState<string>('created_at');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Check auth and admin status
  useEffect(() => {
    if (!authLoading) {
      if (!isAuthenticated) {
        router.push('/login');
      } else if (!user?.is_admin) {
        router.push('/access-denied');
      }
    }
  }, [authLoading, isAuthenticated, user, router]);

  // Fetch analysis stats
  const { data: analysisStats } = useQuery({
    queryKey: ['analysis-stats'],
    queryFn: () => analysisApi.getStats(),
    enabled: isAuthenticated && user?.is_admin,
  });

  // Fetch users
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['admin-users'],
    queryFn: () => usersApi.getUsers({ per_page: 100 }),
    enabled: isAuthenticated && user?.is_admin,
  });

  // Toggle admin mutation
  const toggleAdminMutation = useMutation({
    mutationFn: (userId: string) => usersApi.toggleAdmin(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
    },
  });

  // Toggle active mutation
  const toggleActiveMutation = useMutation({
    mutationFn: (userId: string) => usersApi.toggleActive(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
    },
  });

  // Delete user mutation
  const deleteUserMutation = useMutation({
    mutationFn: (userId: string) => usersApi.deleteUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
      setShowDeleteModal(false);
      setUserToDelete(null);
    },
  });

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const filteredUsers = data?.users
    ?.filter((u: UserData) => {
      if (!searchQuery) return true;
      const query = searchQuery.toLowerCase();
      return (
        u.username.toLowerCase().includes(query) ||
        u.email.toLowerCase().includes(query) ||
        u.first_name.toLowerCase().includes(query) ||
        u.last_name.toLowerCase().includes(query) ||
        (u.phone && u.phone.toLowerCase().includes(query))
      );
    })
    ?.sort((a: UserData, b: UserData) => {
      let aValue: any = a[sortField as keyof UserData];
      let bValue: any = b[sortField as keyof UserData];

      // Handle null values
      if (aValue === null && bValue === null) return 0;
      if (aValue === null) return sortDirection === 'asc' ? -1 : 1;
      if (bValue === null) return sortDirection === 'asc' ? 1 : -1;

      // Handle string comparisons
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      // Handle date comparisons
      if (sortField === 'created_at') {
        aValue = new Date(aValue).getTime();
        bValue = new Date(bValue).getTime();
      }

      if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

  if (authLoading || !user?.is_admin) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-primary-100 dark:bg-primary-900/30">
              <Users className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
              User Management
            </h1>
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            Manage user accounts, permissions, and access
          </p>
        </div>

        {/* Stats Cards */}
        <div className="mb-8 space-y-6">
          {/* Row 1: Total Users and Total Analyses */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Total Users</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {data?.total || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-primary-100 dark:bg-primary-900/30">
                  <Users className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.12 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Total Analyses</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {analysisStats?.total_analyses || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-blue-100 dark:bg-blue-900/30">
                  <Activity className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
              </div>
            </motion.div>
          </div>

          {/* Row 2: User Statistics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
            {/* Owners */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.12 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Owners</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {data?.users?.filter((u: UserData) => u.is_owner).length || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-yellow-100 dark:bg-yellow-900/30">
                  <Shield className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
                </div>
              </div>
            </motion.div>

            {/* Admins */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.14 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Admins</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {data?.users?.filter((u: UserData) => u.is_admin && !u.is_owner).length || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-purple-100 dark:bg-purple-900/30">
                  <Shield className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
              </div>
            </motion.div>

            {/* Standard Users */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.16 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Standard Users</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {(data?.users?.filter((u: UserData) => !u.is_admin && !u.is_owner).length || 0)}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-900/30">
                  <Users className="w-6 h-6 text-slate-600 dark:text-slate-400" />
                </div>
              </div>
            </motion.div>

            {/* Active */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.18 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Active</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {data?.users?.filter((u: UserData) => u.is_active).length || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-green-100 dark:bg-green-900/30">
                  <UserCheck className="w-6 h-6 text-green-600 dark:text-green-400" />
                </div>
              </div>
            </motion.div>

            {/* Inactive */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Inactive</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {data?.users?.filter((u: UserData) => !u.is_active).length || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-red-100 dark:bg-red-900/30">
                  <UserX className="w-6 h-6 text-red-600 dark:text-red-400" />
                </div>
              </div>
            </motion.div>
          </div>

          {/* Row 3: Analysis Types */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
            {/* Heart Disease */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.22 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Heart Disease</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {analysisStats?.by_type?.['heart-prediction'] || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-red-100 dark:bg-red-900/30">
                  <Heart className="w-6 h-6 text-red-600 dark:text-red-400" />
                </div>
              </div>
            </motion.div>

            {/* Diabetes */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.24 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Diabetes</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {analysisStats?.by_type?.['diabetes-prediction'] || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-orange-100 dark:bg-orange-900/30">
                  <Droplets className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                </div>
              </div>
            </motion.div>

            {/* Skin Diagnosis */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.26 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Skin Analysis</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {analysisStats?.by_type?.['skin-diagnosis'] || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-teal-100 dark:bg-teal-900/30">
                  <Scan className="w-6 h-6 text-teal-600 dark:text-teal-400" />
                </div>
              </div>
            </motion.div>

            {/* Breast Prediction */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.28 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Breast Risk</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {analysisStats?.by_type?.['breast-prediction'] || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-pink-100 dark:bg-pink-900/30">
                  <Ribbon className="w-6 h-6 text-pink-600 dark:text-pink-400" />
                </div>
              </div>
            </motion.div>

            {/* Breast Diagnosis */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Breast Tissue</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {analysisStats?.by_type?.['breast-diagnosis'] || 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-fuchsia-100 dark:bg-fuchsia-900/30">
                  <Ribbon className="w-6 h-6 text-fuchsia-600 dark:text-fuchsia-400" />
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search by username, email, or name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input pl-12 w-full md:w-96"
            />
          </div>
        </div>

        {/* Users Table */}
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 dark:bg-slate-800">
                <tr>
                  <th className="px-6 py-4 text-left">
                    <button
                      onClick={() => handleSort('first_name')}
                      className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider hover:text-slate-700 dark:hover:text-slate-300"
                    >
                      User
                      {sortField === 'first_name' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="w-4 h-4" /> : 
                          <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left">
                    <button
                      onClick={() => handleSort('email')}
                      className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider hover:text-slate-700 dark:hover:text-slate-300"
                    >
                      Contact
                      {sortField === 'email' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="w-4 h-4" /> : 
                          <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left">
                    <button
                      onClick={() => handleSort('is_active')}
                      className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider hover:text-slate-700 dark:hover:text-slate-300"
                    >
                      Status
                      {sortField === 'is_active' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="w-4 h-4" /> : 
                          <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left">
                    <button
                      onClick={() => handleSort('is_admin')}
                      className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider hover:text-slate-700 dark:hover:text-slate-300"
                    >
                      Admin
                      {sortField === 'is_admin' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="w-4 h-4" /> : 
                          <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left">
                    <button
                      onClick={() => handleSort('analysis_count')}
                      className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider hover:text-slate-700 dark:hover:text-slate-300"
                    >
                      Analyses
                      {sortField === 'analysis_count' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="w-4 h-4" /> : 
                          <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-4 text-left">
                    <button
                      onClick={() => handleSort('created_at')}
                      className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider hover:text-slate-700 dark:hover:text-slate-300"
                    >
                      Joined
                      {sortField === 'created_at' && (
                        sortDirection === 'asc' ? 
                          <ChevronUp className="w-4 h-4" /> : 
                          <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                  </th>
                  {user?.is_owner && (
                    <th className="px-6 py-4 text-right text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      Actions
                    </th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                {isLoading ? (
                  <tr>
                    <td colSpan={user?.is_owner ? 7 : 6} className="px-6 py-12 text-center">
                      <div className="flex items-center justify-center gap-3">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500" />
                        <span className="text-slate-500">Loading users...</span>
                      </div>
                    </td>
                  </tr>
                ) : filteredUsers?.length === 0 ? (
                  <tr>
                    <td colSpan={user?.is_owner ? 7 : 6} className="px-6 py-12 text-center text-slate-500">
                      No users found
                    </td>
                  </tr>
                ) : (
                  filteredUsers?.map((u: UserData) => (
                    <tr
                      key={u.id}
                      className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors cursor-pointer group"
                    >
                      <Link href={`/admin/users/${u.id}`} className="contents">
                        <td className="px-6 py-4 align-middle">
                          <div className="flex items-center gap-3">
                            {(u as any).avatar_url ? (
                              <img
                                src={getImageURL((u as any).avatar_url)}
                                alt={`${u.first_name} ${u.last_name}`}
                                className="w-10 h-10 rounded-full object-cover"
                              />
                            ) : (
                              <div className="w-10 h-10 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center flex-shrink-0">
                                <span className="text-primary-600 dark:text-primary-400 font-semibold">
                                  {u.first_name[0]}{u.last_name[0]}
                                </span>
                              </div>
                            )}
                            <div>
                              <p className="font-medium text-slate-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                                {u.first_name} {u.last_name}
                              </p>
                              <div className="flex items-center gap-2">
                                <p className="text-sm text-slate-500 dark:text-slate-400">
                                  @{u.username}
                                </p>
                                {u.id === user?.id && (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                                    <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                                    You
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 align-middle">
                          <div className="space-y-1">
                            <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                              <Mail className="w-4 h-4" />
                              {u.email}
                            </div>
                            {u.phone && (
                              <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                                <Phone className="w-4 h-4" />
                                {u.phone}
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4 align-middle">
                          {u.is_active ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                              <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                              Active
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400">
                              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                              Inactive
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 align-middle">
                          {u.is_owner ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400">
                              <Shield className="w-3 h-3" />
                              Owner
                            </span>
                          ) : u.is_admin ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400">
                              <Shield className="w-3 h-3" />
                              Admin
                            </span>
                          ) : (
                            <span className="text-sm text-slate-400">User</span>
                          )}
                        </td>
                        <td className="px-6 py-4 align-middle">
                          <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-slate-400" />
                            <span className="text-sm font-medium text-slate-900 dark:text-white">
                              {u.analysis_count || 0}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 align-middle">
                          <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                            <Calendar className="w-4 h-4" />
                            {formatDateTime(u.created_at)}
                          </div>
                        </td>
                      </Link>
                      {user?.is_owner && (
                        <td className="px-6 py-4 text-right align-middle">
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleAdminMutation.mutate(u.id);
                              }}
                              disabled={toggleAdminMutation.isPending || u.is_owner}
                              className={`p-2 rounded-lg transition-colors ${
                                u.is_owner 
                                  ? 'text-slate-300 dark:text-slate-600 cursor-not-allowed opacity-50' 
                                  : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                              }`}
                              title={u.is_owner ? "Cannot modify owner permissions" : (u.is_admin ? "Remove Admin" : "Make Admin")}
                            >
                              {u.is_admin ? (
                                <ShieldOff className="w-4 h-4" />
                              ) : (
                                <Shield className="w-4 h-4" />
                              )}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleActiveMutation.mutate(u.id);
                              }}
                              disabled={toggleActiveMutation.isPending || u.is_owner}
                              className={`p-2 rounded-lg transition-colors ${
                                u.is_owner 
                                  ? 'text-slate-300 dark:text-slate-600 cursor-not-allowed opacity-50' 
                                  : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                              }`}
                              title={u.is_owner ? "Cannot deactivate owner" : (u.is_active ? "Deactivate" : "Activate")}
                            >
                              {u.is_active ? (
                                <UserX className="w-4 h-4" />
                              ) : (
                                <UserCheck className="w-4 h-4" />
                              )}
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setUserToDelete(u);
                                setShowDeleteModal(true);
                              }}
                              disabled={u.is_owner}
                              className={`p-2 rounded-lg transition-colors ${
                                u.is_owner 
                                  ? 'text-slate-300 dark:text-slate-600 cursor-not-allowed opacity-50' 
                                  : 'text-red-400 hover:text-red-600 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20'
                              }`}
                              title={u.is_owner ? "Cannot delete owner" : "Delete User"}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </td>
                      )}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && userToDelete && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl max-w-md w-full p-6"
          >
            <div className="flex items-center gap-4 mb-4">
              <div className={`p-3 rounded-full ${
                userToDelete.is_owner 
                  ? 'bg-amber-100 dark:bg-amber-900/30' 
                  : 'bg-red-100 dark:bg-red-900/30'
              }`}>
                <AlertTriangle className={`w-6 h-6 ${
                  userToDelete.is_owner 
                    ? 'text-amber-600 dark:text-amber-400' 
                    : 'text-red-600 dark:text-red-400'
                }`} />
              </div>
              <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                {userToDelete.is_owner ? 'Cannot Delete Owner' : 'Delete User'}
              </h3>
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-6">
              {userToDelete.is_owner ? (
                <>
                  <strong>{userToDelete.first_name} {userToDelete.last_name}</strong> (@{userToDelete.username}) is a system owner and cannot be deleted. 
                  Owner accounts are protected to maintain system integrity.
                </>
              ) : (
                <>
                  Are you sure you want to delete <strong>{userToDelete.first_name} {userToDelete.last_name}</strong> (@{userToDelete.username})? 
                  This action cannot be undone and will remove all their data including analysis history.
                </>
              )}
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowDeleteModal(false);
                  setUserToDelete(null);
                }}
                className="btn btn-outline flex-1"
              >
                {userToDelete.is_owner ? 'Close' : 'Cancel'}
              </button>
              {!userToDelete.is_owner && (
                <button
                  onClick={() => deleteUserMutation.mutate(userToDelete.id)}
                  disabled={deleteUserMutation.isPending}
                  className="btn bg-red-600 hover:bg-red-700 text-white flex-1"
                >
                  {deleteUserMutation.isPending ? 'Deleting...' : 'Delete User'}
                </button>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
